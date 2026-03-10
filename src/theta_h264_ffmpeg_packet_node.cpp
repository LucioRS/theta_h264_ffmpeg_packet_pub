#include <rclcpp/rclcpp.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

using ffmpeg_image_transport_msgs::msg::FFMPEGPacket;
using namespace std::chrono_literals;

namespace
{

bool host_is_big_endian()
{
  const uint16_t x = 0x0102;
  return *(reinterpret_cast<const uint8_t *>(&x)) == 0x01;
}

std::string ff_err_str(int errnum)
{
  char buf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(errnum, buf, sizeof(buf));
  return std::string(buf);
}

std::string pix_fmt_name(AVPixelFormat fmt)
{
  const char * n = av_get_pix_fmt_name(fmt);
  return n ? std::string(n) : std::string("unknown");
}

int64_t steady_time_ns()
{
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string to_upper_ascii(std::string s)
{
  for (char & c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

struct QueuedFrame
{
  std::vector<uint8_t> data;
  size_t bytes_used{0};
  int64_t capture_ns{0};
  builtin_interfaces::msg::Time ros_stamp;
};

struct EncoderIoStats
{
  long long send_us{0};
  long long recv_us{0};
  int packets{0};
};

}  // namespace

class ThetaH264ToFFMPEGPacketNode : public rclcpp::Node
{
public:
  ThetaH264ToFFMPEGPacketNode()
  : Node("theta_h264_ffmpeg_packet_pub")
  {
    gst_video_info_init(&gst_video_info_);

    // Camera / topic
    camera_mode_ = declare_parameter<std::string>("camera_mode", "4K");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera");
    base_topic_ = declare_parameter<std::string>("base_topic", "/camera/image_raw");

    // Encoder
    encoder_name_ = declare_parameter<std::string>("encoder_name", "h264_vaapi");
    bit_rate_ = declare_parameter<int>("bit_rate", 20000000);
    gop_size_ = declare_parameter<int>("gop_size", 10);
    max_b_frames_ = declare_parameter<int>("max_b_frames", 0);
    preset_ = declare_parameter<std::string>("preset", "ultrafast");
    tune_ = declare_parameter<std::string>("tune", "zerolatency");
    crf_ = declare_parameter<double>("crf", -1.0);  // used for libx264 if >= 0

    // Extra VAAPI tuning
    compression_level_ = declare_parameter<int>("compression_level", -1);  // -1 = leave default
    async_depth_ = declare_parameter<int>("async_depth", 4);
    rc_mode_ = declare_parameter<std::string>("rc_mode", "auto");          // "auto" = leave default
    global_quality_ = declare_parameter<int>("global_quality", 0);         // 0 = unset
    low_power_ = declare_parameter<bool>("low_power", false);

    // VAAPI
    vaapi_device_ = declare_parameter<std::string>("vaapi_device", "/dev/dri/renderD128");
    vaapi_pool_size_ = declare_parameter<int>("vaapi_pool_size", 20);

    // Timing / PTS
    pts_timebase_hz_ = declare_parameter<int>("pts_timebase_hz", 1000000);  // microseconds

    // Queueing / buffering
    queue_capacity_ = declare_parameter<int>("queue_capacity", 6);
    drop_oldest_when_queue_full_ =
      declare_parameter<bool>("drop_oldest_when_queue_full", true);

    // QoS
    reliable_qos_ = declare_parameter<bool>("reliable_qos", false);
    qos_depth_ = declare_parameter<int>("qos_depth", 10);

    // GStreamer
    gst_pull_timeout_ms_ = declare_parameter<int>("gst_pull_timeout_ms", 200);
    gst_startup_timeout_ms_ = declare_parameter<int>("gst_startup_timeout_ms", 5000);

    // Debug timing
    enable_timing_logs_ = declare_parameter<bool>("enable_timing_logs", true);
    timing_log_every_n_frames_ =
      declare_parameter<int>("timing_log_every_n_frames", 60);

    // Lifecycle
    watchdog_period_ms_ = declare_parameter<int>("watchdog_period_ms", 100);

    const std::string topic = base_topic_ + "/ffmpeg";

    auto qos = rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qos_depth_)));
    qos.durability_volatile();
    if (reliable_qos_) {
      qos.reliable();
    } else {
      qos.best_effort();
    }

    pub_ = create_publisher<FFMPEGPacket>(topic, qos);

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(watchdog_period_ms_),
      std::bind(&ThetaH264ToFFMPEGPacketNode::watchdog_cb, this));

    control_thread_ = std::thread(&ThetaH264ToFFMPEGPacketNode::control_loop, this);

    RCLCPP_INFO(
      get_logger(),
      "Ready. Will start GStreamer decode + FFmpeg encode only when there is at least one "
      "subscriber on %s",
      topic.c_str());
  }

  ~ThetaH264ToFFMPEGPacketNode() override
  {
    shutdown_requested_.store(true);

    if (watchdog_timer_) {
      watchdog_timer_->cancel();
    }

    {
      std::lock_guard<std::mutex> lock(control_mutex_);
      control_dirty_ = true;
    }
    control_cv_.notify_all();

    if (control_thread_.joinable()) {
      control_thread_.join();
    }
  }

private:
  bool interrupted() const
  {
    return shutdown_requested_.load() || !rclcpp::ok();
  }

  bool using_vaapi() const
  {
    return encoder_name_.find("vaapi") != std::string::npos;
  }

  bool using_libx264() const
  {
    return encoder_name_.find("libx264") != std::string::npos;
  }

  size_t subscriber_count() const
  {
    if (!pub_) {
      return 0;
    }
    return pub_->get_subscription_count() + pub_->get_intra_process_subscription_count();
  }

  void watchdog_cb()
  {
    if (shutdown_requested_) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(control_mutex_);
      desired_running_ = (subscriber_count() > 0);
      control_dirty_ = true;
    }

    control_cv_.notify_one();
  }

  void control_loop()
  {
    while (true) {
      bool should_run = false;

      {
        std::unique_lock<std::mutex> lock(control_mutex_);
        control_cv_.wait(lock, [&]() {
          return control_dirty_ || shutdown_requested_;
        });

        if (shutdown_requested_) {
          break;
        }

        should_run = desired_running_;
        control_dirty_ = false;
      }

      try {
        if (should_run) {
          start_pipeline();
        } else {
          stop_pipeline();
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "control_loop exception: %s", e.what());
      }
    }

    try {
      stop_pipeline();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "stop_pipeline during shutdown failed: %s", e.what());
    }
  }

  std::string current_encoding_string() const
  {
    if (!codec_ctx_) {
      return "h264";
    }

    switch (codec_ctx_->codec_id) {
      case AV_CODEC_ID_H264:
        return "h264";
      case AV_CODEC_ID_HEVC:
        return "hevc";
      default:
        return "h264";
    }
  }

  double nominal_fps() const
  {
    return (fps_den_ > 0) ? static_cast<double>(fps_num_) / static_cast<double>(fps_den_) : 0.0;
  }

  std::string validated_camera_mode() const
  {
    const std::string mode = to_upper_ascii(camera_mode_);
    if (mode != "2K" && mode != "4K") {
      throw std::runtime_error("camera_mode must be either '2K' or '4K'");
    }
    return mode;
  }

  std::string make_gst_pipeline_desc(const std::string & mode) const
  {
    return
      "thetauvcsrc mode=" + mode +
      " ! queue "
      " ! h264parse "
      " ! vah264dec "
      " ! videoconvert "
      " ! video/x-raw,format=NV12 "
      " ! queue "
      " ! appsink name=theta_sink drop=true max-buffers=1 sync=false";
  }

  void start_pipeline()
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (pipeline_active_ || interrupted()) {
      return;
    }

    try {
      RCLCPP_INFO(get_logger(), "start_pipeline: setup_gstreamer()...");
      setup_gstreamer();
      if (interrupted()) {
        throw std::runtime_error("startup interrupted after setup_gstreamer()");
      }

      RCLCPP_INFO(get_logger(), "start_pipeline: setup_encoder()...");
      setup_encoder();
      if (interrupted()) {
        throw std::runtime_error("startup interrupted after setup_encoder()");
      }

      RCLCPP_INFO(get_logger(), "start_pipeline: setup_frame_queue()...");
      setup_frame_queue();

      first_capture_ns_ = -1;
      dropped_capture_frames_ = 0;
      capture_timing_counter_ = 0;
      encode_timing_counter_ = 0;

      {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        encoder_should_run_ = true;
      }
      capture_running_ = true;

      encoder_thread_ = std::thread(&ThetaH264ToFFMPEGPacketNode::encoder_loop, this);
      capture_thread_ = std::thread(&ThetaH264ToFFMPEGPacketNode::capture_loop, this);
      pipeline_active_ = true;

      if (using_vaapi()) {
        RCLCPP_INFO(
          get_logger(),
          "Subscriber detected -> starting GStreamer decode + FFmpeg encode "
          "(mode=%s, stream=%dx%d @ %.3f fps, encoder=%s, vaapi_device=%s, "
          "upload_sw_format=%s, queue_capacity=%d, qos=%s, timing_logs=%s every_n=%d, "
          "compression_level=%d, async_depth=%d, rc_mode=%s, global_quality=%d, low_power=%s)",
          camera_mode_.c_str(), width_, height_, nominal_fps(),
          encoder_name_.c_str(), vaapi_device_.c_str(), pix_fmt_name(sw_submit_fmt_).c_str(),
          queue_capacity_, reliable_qos_ ? "reliable" : "best_effort",
          enable_timing_logs_ ? "true" : "false", timing_log_every_n_frames_,
          compression_level_, async_depth_, rc_mode_.c_str(), global_quality_,
          low_power_ ? "true" : "false");
      } else {
        RCLCPP_INFO(
          get_logger(),
          "Subscriber detected -> starting GStreamer decode + FFmpeg encode "
          "(mode=%s, stream=%dx%d @ %.3f fps, encoder=%s, queue_capacity=%d, qos=%s, "
          "timing_logs=%s every_n=%d, preset=%s, tune=%s, crf=%.2f, bit_rate=%d)",
          camera_mode_.c_str(), width_, height_, nominal_fps(),
          encoder_name_.c_str(), queue_capacity_,
          reliable_qos_ ? "reliable" : "best_effort",
          enable_timing_logs_ ? "true" : "false", timing_log_every_n_frames_,
          preset_.c_str(), tune_.c_str(), crf_, bit_rate_);
      }
    } catch (const std::exception & e) {
      capture_running_ = false;
      {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        encoder_should_run_ = false;
      }
      queue_cv_.notify_all();

      pipeline_active_ = false;
      cleanup_encoder();
      cleanup_gstreamer();
      cleanup_frame_queue();

      RCLCPP_ERROR(get_logger(), "Failed to start pipeline: %s", e.what());
    }
  }

  void stop_pipeline()
  {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!pipeline_active_) {
        return;
      }
      capture_running_ = false;
    }

    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }

    {
      std::lock_guard<std::mutex> qlock(queue_mutex_);
      encoder_should_run_ = false;
    }
    queue_cv_.notify_all();

    if (encoder_thread_.joinable()) {
      encoder_thread_.join();
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      cleanup_encoder();
      cleanup_gstreamer();
      cleanup_frame_queue();
      pipeline_active_ = false;
    }

    RCLCPP_INFO(
      get_logger(),
      "No subscribers -> decode/encode stopped (dropped_capture_frames=%llu)",
      static_cast<unsigned long long>(dropped_capture_frames_.load()));
  }

  void drain_gst_bus_messages(bool throw_on_error)
  {
    if (!gst_bus_) {
      return;
    }

    while (true) {
      GstMessage * msg = gst_bus_pop_filtered(
        gst_bus_,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS));

      if (!msg) {
        break;
      }

      switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_WARNING:
        {
          GError * err = nullptr;
          gchar * dbg = nullptr;
          gst_message_parse_warning(msg, &err, &dbg);
          RCLCPP_WARN(
            get_logger(),
            "GStreamer warning from %s: %s%s%s",
            GST_OBJECT_NAME(msg->src),
            err ? err->message : "unknown warning",
            dbg ? " | debug: " : "",
            dbg ? dbg : "");
          if (err) {
            g_error_free(err);
          }
          if (dbg) {
            g_free(dbg);
          }
          break;
        }

        case GST_MESSAGE_ERROR:
        {
          GError * err = nullptr;
          gchar * dbg = nullptr;
          gst_message_parse_error(msg, &err, &dbg);
          const std::string text =
            std::string("GStreamer error from ") + GST_OBJECT_NAME(msg->src) + ": " +
            (err ? err->message : "unknown error") +
            (dbg ? std::string(" | debug: ") + dbg : std::string());

          if (err) {
            g_error_free(err);
          }
          if (dbg) {
            g_free(dbg);
          }
          gst_message_unref(msg);

          if (throw_on_error) {
            throw std::runtime_error(text);
          } else {
            RCLCPP_ERROR(get_logger(), "%s", text.c_str());
          }
          return;
        }

        case GST_MESSAGE_EOS:
          if (throw_on_error) {
            gst_message_unref(msg);
            throw std::runtime_error("GStreamer pipeline reached EOS");
          } else {
            RCLCPP_WARN(get_logger(), "GStreamer pipeline reached EOS");
          }
          break;

        default:
          break;
      }

      gst_message_unref(msg);
    }
  }

  void extract_stream_info_from_sample(GstSample * sample)
  {
    if (!sample) {
      throw std::runtime_error("Null GstSample");
    }

    GstCaps * caps = gst_sample_get_caps(sample);
    if (!caps) {
      throw std::runtime_error("Decoded sample has no caps");
    }

    GstVideoInfo info;
    gst_video_info_init(&info);

    if (!gst_video_info_from_caps(&info, caps)) {
      throw std::runtime_error("Failed to parse GstVideoInfo from appsink caps");
    }

    if (GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
      throw std::runtime_error(
        std::string("Expected NV12 from appsink, got ") +
        gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&info)));
    }

    width_ = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
    height_ = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
    fps_num_ = static_cast<int>(GST_VIDEO_INFO_FPS_N(&info));
    fps_den_ = static_cast<int>(GST_VIDEO_INFO_FPS_D(&info));

    if (width_ <= 0 || height_ <= 0) {
      throw std::runtime_error("Invalid decoded frame dimensions from appsink caps");
    }
    if ((width_ % 2) != 0 || (height_ % 2) != 0) {
      throw std::runtime_error("NV12 frame dimensions must be even");
    }

    if (fps_num_ <= 0 || fps_den_ <= 0) {
      fps_num_ = 30;
      fps_den_ = 1;
      RCLCPP_WARN(
        get_logger(),
        "Could not read framerate from caps, defaulting internal framerate to %d/%d",
        fps_num_, fps_den_);
    }

    src_stride_y_ = width_;
    src_stride_uv_ = width_;
    src_frame_bytes_ =
      static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3ULL / 2ULL;

    gst_video_info_ = info;
    gst_video_info_valid_ = true;
  }

  void setup_gstreamer()
  {
    static std::once_flag gst_once;
    std::call_once(gst_once, []() {
      gst_init(nullptr, nullptr);
    });

    camera_mode_ = validated_camera_mode();
    gst_pipeline_desc_ = make_gst_pipeline_desc(camera_mode_);

    RCLCPP_INFO(get_logger(), "GStreamer pipeline: %s", gst_pipeline_desc_.c_str());

    GError * error = nullptr;
    pipeline_ = gst_parse_launch(gst_pipeline_desc_.c_str(), &error);
    if (!pipeline_) {
      const std::string msg = error ? error->message : "unknown error";
      if (error) {
        g_error_free(error);
      }
      throw std::runtime_error("gst_parse_launch failed: " + msg);
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "theta_sink");
    if (!appsink_) {
      throw std::runtime_error("Failed to retrieve appsink named 'theta_sink'");
    }

    gst_bus_ = gst_element_get_bus(pipeline_);
    if (!gst_bus_) {
      throw std::runtime_error("Failed to get GStreamer bus");
    }

    gst_app_sink_set_emit_signals(GST_APP_SINK(appsink_), FALSE);
    gst_app_sink_set_drop(GST_APP_SINK(appsink_), TRUE);
    gst_app_sink_set_max_buffers(GST_APP_SINK(appsink_), 1);
    g_object_set(G_OBJECT(appsink_), "sync", FALSE, nullptr);

    GstCaps * sink_caps = gst_caps_from_string("video/x-raw,format=NV12");
    gst_app_sink_set_caps(GST_APP_SINK(appsink_), sink_caps);
    gst_caps_unref(sink_caps);

    if (interrupted()) {
      throw std::runtime_error("GStreamer startup interrupted before PLAYING");
    }

    const GstStateChangeReturn sret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (sret == GST_STATE_CHANGE_FAILURE) {
      throw std::runtime_error("Failed to set GStreamer pipeline to PLAYING");
    }

    GstState state = GST_STATE_NULL;
    GstState pending = GST_STATE_NULL;
    const auto state_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(gst_startup_timeout_ms_);

    bool state_ready = false;
    while (!interrupted() && std::chrono::steady_clock::now() < state_deadline) {
      const GstStateChangeReturn gret = gst_element_get_state(
        pipeline_, &state, &pending, 100 * GST_MSECOND);

      if (gret == GST_STATE_CHANGE_FAILURE) {
        drain_gst_bus_messages(true);
        throw std::runtime_error("GStreamer pipeline failed while waiting for PLAYING state");
      }

      if (gret == GST_STATE_CHANGE_SUCCESS ||
          gret == GST_STATE_CHANGE_NO_PREROLL ||
          state == GST_STATE_PLAYING)
      {
        state_ready = true;
        break;
      }

      drain_gst_bus_messages(false);
    }

    if (interrupted()) {
      throw std::runtime_error("GStreamer startup interrupted");
    }

    if (!state_ready) {
      drain_gst_bus_messages(true);
      throw std::runtime_error("Timed out waiting for GStreamer PLAYING state");
    }

    const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(gst_startup_timeout_ms_);

    while (!interrupted() && std::chrono::steady_clock::now() < deadline) {
      drain_gst_bus_messages(false);

      GstSample * sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(appsink_),
        100 * GST_MSECOND);

      if (sample) {
        extract_stream_info_from_sample(sample);
        gst_sample_unref(sample);
        return;
      }

      if (gst_app_sink_is_eos(GST_APP_SINK(appsink_))) {
        throw std::runtime_error("GStreamer appsink reached EOS before first sample");
      }
    }

    if (interrupted()) {
      throw std::runtime_error("GStreamer startup interrupted");
    }

    drain_gst_bus_messages(true);
    throw std::runtime_error("Timed out waiting for first decoded sample from appsink");
  }

  void cleanup_gstreamer()
  {
    if (pipeline_) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
    }

    if (gst_bus_) {
      gst_object_unref(gst_bus_);
      gst_bus_ = nullptr;
    }

    if (appsink_) {
      gst_object_unref(appsink_);
      appsink_ = nullptr;
    }

    if (pipeline_) {
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
    }

    gst_video_info_init(&gst_video_info_);
    gst_video_info_valid_ = false;
    gst_pipeline_desc_.clear();

    width_ = 0;
    height_ = 0;
    fps_num_ = 30;
    fps_den_ = 1;
    src_stride_y_ = 0;
    src_stride_uv_ = 0;
    src_frame_bytes_ = 0;
  }

  void setup_frame_queue()
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    frame_pool_.clear();
    free_slots_.clear();
    ready_slots_.clear();

    frame_pool_.resize(static_cast<size_t>(queue_capacity_));
    for (size_t i = 0; i < frame_pool_.size(); ++i) {
      frame_pool_[i].data.resize(src_frame_bytes_);
      frame_pool_[i].bytes_used = 0;
      frame_pool_[i].capture_ns = 0;
      free_slots_.push_back(i);
    }
  }

  void cleanup_frame_queue()
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_pool_.clear();
    free_slots_.clear();
    ready_slots_.clear();
  }

  AVFrame * alloc_sw_frame(AVPixelFormat fmt)
  {
    AVFrame * f = av_frame_alloc();
    if (!f) {
      throw std::runtime_error("av_frame_alloc() failed");
    }

    f->format = fmt;
    f->width = width_;
    f->height = height_;

    const int ret = av_frame_get_buffer(f, 32);
    if (ret < 0) {
      av_frame_free(&f);
      throw std::runtime_error("av_frame_get_buffer() failed: " + ff_err_str(ret));
    }

    return f;
  }

  void setup_vaapi_encoder()
  {
    int ret = av_hwdevice_ctx_create(
      &hw_device_ctx_,
      AV_HWDEVICE_TYPE_VAAPI,
      vaapi_device_.empty() ? nullptr : vaapi_device_.c_str(),
      nullptr,
      0);

    if (ret < 0) {
      throw std::runtime_error(
        "av_hwdevice_ctx_create(VAAPI) failed: " + ff_err_str(ret));
    }

    const AVCodec * codec = avcodec_find_encoder_by_name(encoder_name_.c_str());
    if (!codec) {
      throw std::runtime_error("Could not find encoder: " + encoder_name_);
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      throw std::runtime_error("avcodec_alloc_context3() failed");
    }

    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx_->codec_id = codec->id;
    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->pix_fmt = AV_PIX_FMT_VAAPI;
    codec_ctx_->time_base = AVRational{1, pts_timebase_hz_};
    codec_ctx_->framerate = AVRational{fps_num_, fps_den_};
    codec_ctx_->bit_rate = bit_rate_;
    codec_ctx_->gop_size = gop_size_;
    codec_ctx_->max_b_frames = max_b_frames_;
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

    if (compression_level_ >= 0) {
      codec_ctx_->compression_level = compression_level_;
    }
    if (global_quality_ > 0) {
      codec_ctx_->global_quality = global_quality_;
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    if (!codec_ctx_->hw_device_ctx) {
      throw std::runtime_error("av_buffer_ref(hw_device_ctx_) failed");
    }

    if (codec_ctx_->priv_data) {
      av_opt_set(codec_ctx_->priv_data, "preset", preset_.c_str(), 0);
      av_opt_set(codec_ctx_->priv_data, "tune", tune_.c_str(), 0);
      av_opt_set_int(codec_ctx_->priv_data, "async_depth", async_depth_, 0);
      av_opt_set_int(codec_ctx_->priv_data, "low_power", low_power_ ? 1 : 0, 0);

      if (!rc_mode_.empty() && rc_mode_ != "auto") {
        av_opt_set(codec_ctx_->priv_data, "rc_mode", rc_mode_.c_str(), 0);
      }
    }

    hw_frames_ref_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ref_) {
      throw std::runtime_error("av_hwframe_ctx_alloc() failed");
    }

    {
      auto * frames_ctx = reinterpret_cast<AVHWFramesContext *>(hw_frames_ref_->data);
      frames_ctx->format = AV_PIX_FMT_VAAPI;
      frames_ctx->sw_format = AV_PIX_FMT_NV12;
      frames_ctx->width = width_;
      frames_ctx->height = height_;
      frames_ctx->initial_pool_size = vaapi_pool_size_;
    }

    ret = av_hwframe_ctx_init(hw_frames_ref_);
    if (ret < 0) {
      throw std::runtime_error(
        "av_hwframe_ctx_init(NV12) failed: " + ff_err_str(ret));
    }

    codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ref_);
    if (!codec_ctx_->hw_frames_ctx) {
      throw std::runtime_error("av_buffer_ref(hw_frames_ref_) failed");
    }

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
      throw std::runtime_error(
        "avcodec_open2() failed for encoder " + encoder_name_ + ": " + ff_err_str(ret));
    }

    sw_submit_fmt_ = AV_PIX_FMT_NV12;
    sw_submit_frame_ = alloc_sw_frame(sw_submit_fmt_);

    hw_frame_ = av_frame_alloc();
    if (!hw_frame_) {
      throw std::runtime_error("av_frame_alloc() failed for hw_frame_");
    }

    packet_ = av_packet_alloc();
    if (!packet_) {
      throw std::runtime_error("av_packet_alloc() failed");
    }
  }

  void setup_software_encoder()
  {
    const AVCodec * codec = avcodec_find_encoder_by_name(encoder_name_.c_str());
    if (!codec) {
      throw std::runtime_error("Could not find encoder: " + encoder_name_);
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      throw std::runtime_error("avcodec_alloc_context3() failed");
    }

    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx_->codec_id = codec->id;
    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx_->time_base = AVRational{1, pts_timebase_hz_};
    codec_ctx_->framerate = AVRational{fps_num_, fps_den_};
    codec_ctx_->gop_size = gop_size_;
    codec_ctx_->max_b_frames = max_b_frames_;
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

    const bool use_crf_for_x264 = using_libx264() && crf_ >= 0.0;
    if (use_crf_for_x264) {
      codec_ctx_->bit_rate = 0;
    } else {
      codec_ctx_->bit_rate = bit_rate_;
    }

    if (codec_ctx_->priv_data) {
      av_opt_set(codec_ctx_->priv_data, "preset", preset_.c_str(), 0);
      av_opt_set(codec_ctx_->priv_data, "tune", tune_.c_str(), 0);

      if (use_crf_for_x264) {
        av_opt_set_double(codec_ctx_->priv_data, "crf", crf_, 0);
      }
    }

    const int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
      throw std::runtime_error(
        "avcodec_open2() failed for encoder " + encoder_name_ + ": " + ff_err_str(ret));
    }

    sw_submit_fmt_ = AV_PIX_FMT_YUV420P;
    sw_submit_frame_ = alloc_sw_frame(sw_submit_fmt_);

    packet_ = av_packet_alloc();
    if (!packet_) {
      throw std::runtime_error("av_packet_alloc() failed");
    }
  }

  void setup_encoder()
  {
    if (using_vaapi()) {
      setup_vaapi_encoder();
    } else {
      setup_software_encoder();
    }
  }

  void cleanup_encoder()
  {
    if (codec_ctx_) {
      avcodec_send_frame(codec_ctx_, nullptr);
      while (packet_ && avcodec_receive_packet(codec_ctx_, packet_) == 0) {
        av_packet_unref(packet_);
      }
    }

    if (packet_) {
      av_packet_free(&packet_);
      packet_ = nullptr;
    }

    if (hw_frame_) {
      av_frame_free(&hw_frame_);
      hw_frame_ = nullptr;
    }

    if (sw_submit_frame_) {
      av_frame_free(&sw_submit_frame_);
      sw_submit_frame_ = nullptr;
    }

    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
      codec_ctx_ = nullptr;
    }

    if (hw_frames_ref_) {
      av_buffer_unref(&hw_frames_ref_);
      hw_frames_ref_ = nullptr;
    }

    if (hw_device_ctx_) {
      av_buffer_unref(&hw_device_ctx_);
      hw_device_ctx_ = nullptr;
    }

    sw_submit_fmt_ = AV_PIX_FMT_NONE;
  }

  void pack_nv12_to_yuv420p_frame(const uint8_t * src, size_t src_bytes, AVFrame * dst, int64_t pts)
  {
    if (src_bytes < src_frame_bytes_) {
      throw std::runtime_error(
        "Short NV12 frame received: got " + std::to_string(src_bytes) +
        ", expected at least " + std::to_string(src_frame_bytes_));
    }

    const int ret = av_frame_make_writable(dst);
    if (ret < 0) {
      throw std::runtime_error("av_frame_make_writable(YUV420P) failed: " + ff_err_str(ret));
    }

    const uint8_t * src_y = src;
    const uint8_t * src_uv =
      src + static_cast<size_t>(src_stride_y_) * static_cast<size_t>(height_);

    for (int y = 0; y < height_; ++y) {
      std::memcpy(
        dst->data[0] + y * dst->linesize[0],
        src_y + static_cast<size_t>(y) * static_cast<size_t>(src_stride_y_),
        static_cast<size_t>(width_));
    }

    for (int y = 0; y < height_ / 2; ++y) {
      const uint8_t * row_uv =
        src_uv + static_cast<size_t>(y) * static_cast<size_t>(src_stride_uv_);

      uint8_t * dst_u = dst->data[1] + y * dst->linesize[1];
      uint8_t * dst_v = dst->data[2] + y * dst->linesize[2];

      for (int x = 0; x < width_ / 2; ++x) {
        dst_u[x] = row_uv[2 * x + 0];
        dst_v[x] = row_uv[2 * x + 1];
      }
    }

    dst->pts = pts;
  }

  void pack_nv12_to_nv12_frame(const uint8_t * src, size_t src_bytes, AVFrame * dst, int64_t pts)
  {
    if (src_bytes < src_frame_bytes_) {
      throw std::runtime_error(
        "Short NV12 frame received: got " + std::to_string(src_bytes) +
        ", expected at least " + std::to_string(src_frame_bytes_));
    }

    const int ret = av_frame_make_writable(dst);
    if (ret < 0) {
      throw std::runtime_error("av_frame_make_writable(NV12) failed: " + ff_err_str(ret));
    }

    const uint8_t * src_y = src;
    const uint8_t * src_uv =
      src + static_cast<size_t>(src_stride_y_) * static_cast<size_t>(height_);

    for (int y = 0; y < height_; ++y) {
      std::memcpy(
        dst->data[0] + y * dst->linesize[0],
        src_y + static_cast<size_t>(y) * static_cast<size_t>(src_stride_y_),
        static_cast<size_t>(width_));
    }

    for (int y = 0; y < height_ / 2; ++y) {
      std::memcpy(
        dst->data[1] + y * dst->linesize[1],
        src_uv + static_cast<size_t>(y) * static_cast<size_t>(src_stride_uv_),
        static_cast<size_t>(width_));
    }

    dst->pts = pts;
  }

  int64_t capture_ns_to_pts(int64_t capture_ns)
  {
    if (first_capture_ns_ < 0) {
      first_capture_ns_ = capture_ns;
    }
    const int64_t rel_ns = capture_ns - first_capture_ns_;
    return av_rescale_q(
      rel_ns,
      AVRational{1, 1000000000},
      codec_ctx_->time_base);
  }

  void publish_current_packet(const builtin_interfaces::msg::Time & stamp)
  {
    FFMPEGPacket msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id_;
    msg.width = width_;
    msg.height = height_;
    msg.encoding = current_encoding_string();
    msg.pts = (packet_->pts != AV_NOPTS_VALUE) ?
      static_cast<uint64_t>(packet_->pts) : 0ULL;
    msg.flags = static_cast<uint8_t>(packet_->flags & 0xFF);
    msg.is_bigendian = host_is_big_endian();
    msg.data.assign(packet_->data, packet_->data + packet_->size);

    pub_->publish(std::move(msg));
  }

  EncoderIoStats drain_encoder_packets_timed(const builtin_interfaces::msg::Time & stamp)
  {
    using clock = std::chrono::steady_clock;
    EncoderIoStats stats;

    const auto t0 = clock::now();
    while (true) {
      const int ret = avcodec_receive_packet(codec_ctx_, packet_);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      if (ret < 0) {
        throw std::runtime_error("avcodec_receive_packet failed: " + ff_err_str(ret));
      }

      if (subscriber_count() > 0) {
        publish_current_packet(stamp);
      }

      stats.packets++;
      av_packet_unref(packet_);
    }
    const auto t1 = clock::now();
    stats.recv_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return stats;
  }

  EncoderIoStats send_frame_and_drain_timed(AVFrame * frame, const builtin_interfaces::msg::Time & stamp)
  {
    using clock = std::chrono::steady_clock;
    EncoderIoStats total_stats;

    auto send_once = [&](AVFrame * f) -> int {
      const auto t0 = clock::now();
      const int ret = avcodec_send_frame(codec_ctx_, f);
      const auto t1 = clock::now();
      total_stats.send_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      return ret;
    };

    int ret = send_once(frame);

    if (ret == AVERROR(EAGAIN)) {
      EncoderIoStats pre = drain_encoder_packets_timed(stamp);
      total_stats.recv_us += pre.recv_us;
      total_stats.packets += pre.packets;

      ret = send_once(frame);
    }

    if (ret < 0) {
      throw std::runtime_error("avcodec_send_frame failed: " + ff_err_str(ret));
    }

    EncoderIoStats post = drain_encoder_packets_timed(stamp);
    total_stats.recv_us += post.recv_us;
    total_stats.packets += post.packets;

    return total_stats;
  }

  bool should_log_timing(uint64_t & counter) const
  {
    if (!enable_timing_logs_ || timing_log_every_n_frames_ <= 0) {
      return false;
    }
    counter++;
    return (counter % static_cast<uint64_t>(timing_log_every_n_frames_)) == 0;
  }

  size_t ready_queue_size() const
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return ready_slots_.size();
  }

  size_t free_queue_size() const
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return free_slots_.size();
  }

  void encode_one_frame(const QueuedFrame & qf)
  {
    using clock = std::chrono::steady_clock;

    const auto t0 = clock::now();
    const int64_t pts = capture_ns_to_pts(qf.capture_ns);

    long long pack_us = 0;
    long long upload_us = 0;
    long long send_us = 0;
    long long recv_us = 0;
    long long total_us = 0;
    int packets = 0;

    if (using_vaapi()) {
      const auto t_pack0 = clock::now();
      pack_nv12_to_nv12_frame(qf.data.data(), qf.bytes_used, sw_submit_frame_, pts);
      const auto t_pack1 = clock::now();

      av_frame_unref(hw_frame_);

      int ret = av_hwframe_get_buffer(codec_ctx_->hw_frames_ctx, hw_frame_, 0);
      if (ret < 0) {
        throw std::runtime_error("av_hwframe_get_buffer failed: " + ff_err_str(ret));
      }

      ret = av_hwframe_transfer_data(hw_frame_, sw_submit_frame_, 0);
      if (ret < 0) {
        throw std::runtime_error("av_hwframe_transfer_data failed: " + ff_err_str(ret));
      }

      hw_frame_->pts = pts;
      const auto t_upload1 = clock::now();

      EncoderIoStats io = send_frame_and_drain_timed(hw_frame_, qf.ros_stamp);
      const auto t_encode1 = clock::now();

      pack_us = std::chrono::duration_cast<std::chrono::microseconds>(t_pack1 - t_pack0).count();
      upload_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_upload1 - t_pack1).count();
      send_us = io.send_us;
      recv_us = io.recv_us;
      packets = io.packets;
      total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_encode1 - t0).count();
    } else {
      const auto t_pack0 = clock::now();
      pack_nv12_to_yuv420p_frame(qf.data.data(), qf.bytes_used, sw_submit_frame_, pts);
      const auto t_pack1 = clock::now();

      EncoderIoStats io = send_frame_and_drain_timed(sw_submit_frame_, qf.ros_stamp);
      const auto t_encode1 = clock::now();

      pack_us = std::chrono::duration_cast<std::chrono::microseconds>(t_pack1 - t_pack0).count();
      upload_us = 0;
      send_us = io.send_us;
      recv_us = io.recv_us;
      packets = io.packets;
      total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_encode1 - t0).count();
    }

    if (should_log_timing(encode_timing_counter_)) {
      if (using_vaapi()) {
        RCLCPP_INFO(
          get_logger(),
          "[ENC] pack=%lld us upload=%lld us send_frame=%lld us receive_loop=%lld us packets=%d total=%lld us "
          "queue_ready=%zu dropped=%llu",
          pack_us, upload_us, send_us, recv_us, packets, total_us,
          ready_queue_size(),
          static_cast<unsigned long long>(dropped_capture_frames_.load()));
      } else {
        RCLCPP_INFO(
          get_logger(),
          "[ENC] pack=%lld us send_frame=%lld us receive_loop=%lld us packets=%d total=%lld us "
          "queue_ready=%zu dropped=%llu",
          pack_us, send_us, recv_us, packets, total_us,
          ready_queue_size(),
          static_cast<unsigned long long>(dropped_capture_frames_.load()));
      }
    }
  }

  bool enqueue_captured_nv12_frame(
    const GstVideoFrame & frame,
    int64_t capture_ns,
    const builtin_interfaces::msg::Time & ros_stamp)
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    ssize_t slot_idx = -1;

    if (!free_slots_.empty()) {
      slot_idx = static_cast<ssize_t>(free_slots_.front());
      free_slots_.pop_front();
    } else if (drop_oldest_when_queue_full_ && !ready_slots_.empty()) {
      slot_idx = static_cast<ssize_t>(ready_slots_.front());
      ready_slots_.pop_front();
      dropped_capture_frames_++;
    } else {
      dropped_capture_frames_++;
      return false;
    }

    auto & slot = frame_pool_[static_cast<size_t>(slot_idx)];

    uint8_t * dst_y = slot.data.data();
    uint8_t * dst_uv = dst_y + static_cast<size_t>(src_stride_y_) * static_cast<size_t>(height_);

    const auto * src_y =
      reinterpret_cast<const uint8_t *>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
    const auto * src_uv =
      reinterpret_cast<const uint8_t *>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));

    const int gst_stride_y = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
    const int gst_stride_uv = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);

    if (!src_y || !src_uv) {
      throw std::runtime_error("Mapped GstVideoFrame has null plane pointer(s)");
    }

    for (int y = 0; y < height_; ++y) {
      std::memcpy(
        dst_y + static_cast<size_t>(y) * static_cast<size_t>(src_stride_y_),
        src_y + static_cast<size_t>(y) * static_cast<size_t>(gst_stride_y),
        static_cast<size_t>(width_));
    }

    for (int y = 0; y < height_ / 2; ++y) {
      std::memcpy(
        dst_uv + static_cast<size_t>(y) * static_cast<size_t>(src_stride_uv_),
        src_uv + static_cast<size_t>(y) * static_cast<size_t>(gst_stride_uv),
        static_cast<size_t>(width_));
    }

    slot.bytes_used = src_frame_bytes_;
    slot.capture_ns = capture_ns;
    slot.ros_stamp = ros_stamp;

    ready_slots_.push_back(static_cast<size_t>(slot_idx));
    lock.unlock();
    queue_cv_.notify_one();
    return true;
  }

  void release_slot(size_t slot_idx)
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    free_slots_.push_back(slot_idx);
  }

  void capture_loop()
  {
    using clock = std::chrono::steady_clock;

    while (rclcpp::ok() && capture_running_) {
      if (subscriber_count() == 0) {
        std::this_thread::sleep_for(10ms);
        continue;
      }

      GstSample * sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(appsink_),
        static_cast<GstClockTime>(std::max(1, gst_pull_timeout_ms_)) * GST_MSECOND);

      if (!capture_running_) {
        break;
      }

      if (!sample) {
        drain_gst_bus_messages(false);
        if (gst_app_sink_is_eos(GST_APP_SINK(appsink_))) {
          RCLCPP_WARN(get_logger(), "appsink EOS detected, stopping capture loop");
          break;
        }
        continue;
      }

      GstBuffer * buffer = gst_sample_get_buffer(sample);
      if (!buffer) {
        gst_sample_unref(sample);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "GStreamer sample without GstBuffer");
        continue;
      }

      if (!gst_video_info_valid_) {
        gst_sample_unref(sample);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "GstVideoInfo not initialized");
        continue;
      }

      GstVideoFrame vf;
      if (!gst_video_frame_map(&vf, &gst_video_info_, buffer, GST_MAP_READ)) {
        gst_sample_unref(sample);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "gst_video_frame_map() failed");
        continue;
      }

      const auto t_copy0 = clock::now();
      const auto capture_ns = steady_time_ns();

      const auto now_time = now();
      builtin_interfaces::msg::Time ros_stamp;
      {
        const int64_t ns = now_time.nanoseconds();
        ros_stamp.sec = static_cast<int32_t>(ns / 1000000000LL);
        ros_stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
      }

      bool queued = false;
      bool enqueue_attempted = false;

      try {
        if (subscriber_count() > 0) {
          enqueue_attempted = true;
          queued = enqueue_captured_nv12_frame(vf, capture_ns, ros_stamp);
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Capture-side frame enqueue failed: %s", e.what());
      }

      gst_video_frame_unmap(&vf);

      const size_t bytes = gst_buffer_get_size(buffer);
      gst_sample_unref(sample);

      const auto t_copy1 = clock::now();

      if (enqueue_attempted && !queued) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Frame dropped on capture side (queue full). total_dropped=%llu",
          static_cast<unsigned long long>(dropped_capture_frames_.load()));
      }

      if (should_log_timing(capture_timing_counter_)) {
        const long long copy_enqueue_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t_copy1 - t_copy0).count();

        RCLCPP_INFO(
          get_logger(),
          "[CAP] copy+enqueue=%lld us bytes=%zu queued=%s queue_ready=%zu queue_free=%zu dropped=%llu",
          copy_enqueue_us,
          bytes,
          queued ? "true" : "false",
          ready_queue_size(),
          free_queue_size(),
          static_cast<unsigned long long>(dropped_capture_frames_.load()));
      }
    }
  }

  void encoder_loop()
  {
    while (rclcpp::ok()) {
      size_t slot_idx = 0;

      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [&]() {
          return !ready_slots_.empty() || !encoder_should_run_;
        });

        if (ready_slots_.empty() && !encoder_should_run_) {
          break;
        }

        if (ready_slots_.empty()) {
          continue;
        }

        slot_idx = ready_slots_.front();
        ready_slots_.pop_front();
      }

      try {
        encode_one_frame(frame_pool_[slot_idx]);
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Encoder-side frame processing failed: %s", e.what());
      }

      release_slot(slot_idx);
    }
  }

private:
  // Parameters
  std::string camera_mode_{"4K"};
  std::string frame_id_{"camera"};
  std::string base_topic_{"/camera/image_raw"};

  std::string encoder_name_{"h264_vaapi"};
  int bit_rate_{20000000};
  int gop_size_{10};
  int max_b_frames_{0};
  std::string preset_{"ultrafast"};
  std::string tune_{"zerolatency"};
  double crf_{-1.0};

  int compression_level_{-1};
  int async_depth_{4};
  std::string rc_mode_{"auto"};
  int global_quality_{0};
  bool low_power_{false};

  std::string vaapi_device_{"/dev/dri/renderD128"};
  int vaapi_pool_size_{20};

  int pts_timebase_hz_{1000000};

  int queue_capacity_{6};
  bool drop_oldest_when_queue_full_{true};

  bool reliable_qos_{false};
  int qos_depth_{10};

  int gst_pull_timeout_ms_{200};
  int gst_startup_timeout_ms_{5000};

  bool enable_timing_logs_{true};
  int timing_log_every_n_frames_{60};

  int watchdog_period_ms_{100};

  // Derived stream info from appsink caps
  int width_{0};
  int height_{0};
  int fps_num_{30};
  int fps_den_{1};

  // Internal compact NV12 queue layout
  int src_stride_y_{0};
  int src_stride_uv_{0};
  size_t src_frame_bytes_{0};

  // ROS
  rclcpp::Publisher<FFMPEGPacket>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  // Control thread
  std::thread control_thread_;
  std::mutex control_mutex_;
  std::condition_variable control_cv_;
  bool desired_running_{false};
  bool control_dirty_{false};
  std::atomic<bool> shutdown_requested_{false};

  // Lifecycle state
  std::mutex state_mutex_;
  std::atomic<bool> capture_running_{false};
  bool pipeline_active_{false};
  std::thread capture_thread_;
  std::thread encoder_thread_;

  // Queue state
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  bool encoder_should_run_{false};
  std::vector<QueuedFrame> frame_pool_;
  std::deque<size_t> free_slots_;
  std::deque<size_t> ready_slots_;
  std::atomic<uint64_t> dropped_capture_frames_{0};

  // GStreamer
  std::string gst_pipeline_desc_;
  GstElement * pipeline_{nullptr};
  GstElement * appsink_{nullptr};
  GstBus * gst_bus_{nullptr};
  GstVideoInfo gst_video_info_;
  bool gst_video_info_valid_{false};

  // FFmpeg generic
  AVCodecContext * codec_ctx_{nullptr};
  AVPacket * packet_{nullptr};
  AVFrame * sw_submit_frame_{nullptr};
  AVPixelFormat sw_submit_fmt_{AV_PIX_FMT_NONE};

  // VAAPI-specific
  AVBufferRef * hw_device_ctx_{nullptr};
  AVBufferRef * hw_frames_ref_{nullptr};
  AVFrame * hw_frame_{nullptr};

  // Timing
  int64_t first_capture_ns_{-1};
  mutable uint64_t capture_timing_counter_{0};
  mutable uint64_t encode_timing_counter_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ThetaH264ToFFMPEGPacketNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "Fatal error: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}