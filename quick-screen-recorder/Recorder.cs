using NAudio.Wave;
using SharpAvi;
using SharpAvi.Codecs;
using SharpAvi.Output;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using sharpCV = SharpCV;
using System.Reflection;
using System.Drawing;
using Tensorflow;
using System.IO;
using NumSharp;
using Console = Colorful.Console;
using static Tensorflow.Binding;
using static SharpCV.Binding;
using System.Linq;
using CsvHelper;
using System.Globalization;
using System.Collections.Generic;

namespace quick_screen_recorder
{
    class Recorder : IDisposable
    {
        private AviWriter writer;
        private IAviVideoStream videoStream;
        private IAviAudioStream audioStream;

        private IWaveIn audioSource;

        private Thread screenThread;

        private ManualResetEvent stopThread = new ManualResetEvent(false);
        private AutoResetEvent videoFrameWritten = new AutoResetEvent(false);
        private AutoResetEvent audioBlockWritten = new AutoResetEvent(false);

        private WaveFileWriter waveFile;

        private int x;
        private int y;
        private int width;
        private int height;
        private bool captureCursor;

        [StructLayout(LayoutKind.Sequential)]
        public struct POINTAPI
        {
            public int x;
            public int y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct CURSORINFO
        {
            public Int32 cbSize;
            public Int32 flags;
            public IntPtr hCursor;
            public POINTAPI ptScreenPos;
        }

        [DllImport("user32.dll")]
        public static extern bool GetCursorInfo(out CURSORINFO pci);

        [DllImport("user32.dll")]
        public static extern bool DrawIcon(IntPtr hDC, int X, int Y, IntPtr hIcon);

        public const Int32 CURSOR_SHOWING = 0x00000001;

        public bool Mute = false;

        string modelDir = "Models";
        string imageDir = "Inputs";
        string pbDetectorFile = "optimized_frozen_graph.pb";
        string pbEmotionFile = "emotion_frozen_graph.pb";

        int inputSizeDetector = 640;
        int inputSizeEmotion = 60;

        Graph detectorGraph;
        Graph emotionGraph;
        Graph imgGraph;

        Session tfSession;

        Session imgSess;
        Session detectorSession;

        StreamWriter writerCSV;
        CsvWriter csv;
 
        public Recorder(string filePath, 
            int quality, int x, int y, int width, int height, bool captureCursor,
            int inputSourceIndex, bool separateAudio)
        {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.captureCursor = captureCursor;

            writer = new AviWriter(filePath)
            {
                FramesPerSecond = 15,
                EmitIndex1 = true,
            };

            if (quality == 0)
            {
                videoStream = writer.AddUncompressedVideoStream(width, height);
                videoStream.Name = "Quick Screen Recorder - Motion JPEG video stream";
            }
            else
            {
                videoStream = writer.AddMotionJpegVideoStream(width, height, quality);
                videoStream.Name = "Quick Screen Recorder - Motion JPEG video stream";
            }

            if (inputSourceIndex >= 0)
            {
                var waveFormat = new WaveFormat(44100, 16, 1);

                audioStream = writer.AddAudioStream(
                    channelCount: waveFormat.Channels,
                    samplesPerSecond: waveFormat.SampleRate,
                    bitsPerSample: waveFormat.BitsPerSample
                );
                audioStream.Name = "Quick Screen Recorder - Input audio stream";

                audioSource = new WaveInEvent()
                {
                    DeviceNumber = inputSourceIndex,
                    WaveFormat = waveFormat,
                    BufferMilliseconds = (int)Math.Ceiling(1000 / writer.FramesPerSecond),
                    NumberOfBuffers = 3,
                };
                audioSource.DataAvailable += audioSource_DataAvailable;

                if (separateAudio)
                {
                    waveFile = new WaveFileWriter(filePath + " (Input audio).wav", audioSource.WaveFormat);
                }
            } 
            else if (inputSourceIndex == -1)
            {
                audioSource = new WasapiLoopbackCapture();

                audioStream = writer.AddAudioStream(
                    channelCount: 1,
                    samplesPerSecond: audioSource.WaveFormat.SampleRate,
                    bitsPerSample: audioSource.WaveFormat.BitsPerSample
                );
                audioStream.Name = "Quick Screen Recorder - System sounds audio stream";

                audioSource.DataAvailable += audioSource_DataAvailable;

                if (separateAudio)
                {
                    waveFile = new WaveFileWriter(filePath + " (System sounds).wav", audioSource.WaveFormat);
                }
            }

            screenThread = new Thread(RecordScreen)
            {
                Name = typeof(Recorder).Name + ".RecordScreen",
                IsBackground = true
            };

            if (audioSource != null)
            {
                videoFrameWritten.Set();
                audioBlockWritten.Reset();
                audioSource.StartRecording();
            }

            tf.compat.v1.disable_eager_execution();

            //var tfSession = tf.Session();

            detectorGraph = tf.Graph().as_default();
            
            detectorGraph.Import(Path.Combine(modelDir, pbDetectorFile));

            emotionGraph = tf.Graph().as_default();
            emotionGraph.Import(Path.Combine(modelDir, pbEmotionFile));

            Console.WriteLine(Environment.OSVersion, Color.Yellow);
            Console.WriteLine($"64Bit Operating System: {Environment.Is64BitOperatingSystem}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($".NET CLR: {Environment.Version}", Color.Yellow);
            Console.WriteLine(Environment.CurrentDirectory, Color.Yellow);

            //writerCSV = new StreamWriter("output.csv");
            //csv = new CsvWriter(writerCSV, CultureInfo.InvariantCulture);
            using (var writer = new StreamWriter("output.csv"))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteHeader<CSVRecord>();
                csv.NextRecord();
            }

            screenThread.Start();
        }

        public void Dispose()
        {
            stopThread.Set();
            screenThread.Join();

            detectorGraph.Dispose();
            emotionGraph.Dispose();

            if (waveFile != null)
            {
                waveFile.Dispose();
                waveFile = null;
            }

            if (audioSource != null)
            {
                audioSource.StopRecording();
                audioSource.DataAvailable -= audioSource_DataAvailable;
            }

            writer.Close();

            stopThread.Dispose();
        }

        private void RecordScreen()
        {
            var stopwatch = new Stopwatch();
            var buffer = new byte[width * height * 3];

            Task videoWriteTask = null;

            var isFirstFrame = true;
            var shotsTaken = 0;
            var timeTillNextFrame = TimeSpan.Zero;
            stopwatch.Start();

            while (!stopThread.WaitOne(timeTillNextFrame))
            {
                Screenshot(buffer);
                shotsTaken++;
                NDArray img_buffer = new NDArray(buffer);
                img_buffer = img_buffer.reshape((height, width, 3));
                Console.WriteLine($"Image img_buffer: {img_buffer.size}");

                //int x1 = 381;
                //int y1 = 105;
                //int x2 = 644;
                //int y2 = 408;

                //var crop_img = img_buffer[new Slice(y1, y2 + 1), new Slice(x1, x2 + 1), Slice.All];
                //NDArray img_buffer_1 = new NDArray(crop_img.ToArray<byte>());
                //img_buffer_1 = img_buffer_1.reshape((y2 - y1 + 1, x2 - x1 + 1, 3));

                var img_detector = InferenceDetector(img_buffer);
                if (!(img_detector == null))
                {
                    Console.WriteLine($"Image detector: {img_detector.size}");
                    InferenceEmotion(shotsTaken, img_detector);
                }

                if (!isFirstFrame)
                {
                    //videoWriteTask.Wait();

                    //videoFrameWritten.Set();
                }

                if (audioStream != null)
                {
                    //var signalled = WaitHandle.WaitAny(new WaitHandle[] { audioBlockWritten, stopThread });
                    //if (signalled == 1)
                    //    break;
                }

                //videoWriteTask = videoStream.WriteFrameAsync(true, buffer, 0, buffer.Length);

                timeTillNextFrame = TimeSpan.FromSeconds(shotsTaken / (double)writer.FramesPerSecond - stopwatch.Elapsed.TotalSeconds);
                if (timeTillNextFrame < TimeSpan.Zero)
                {
                    timeTillNextFrame = TimeSpan.Zero;
                }

                isFirstFrame = false;
            }

            stopwatch.Stop();

            if (!isFirstFrame)
            {
                //videoWriteTask.Wait();
            }
        }

        private Graph LoadModel(string pbFile)
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Combine(modelDir, pbFile));

            return graph; 
        }

        private NDArray InferenceDetector(NDArray img_buffer_)
        {
            var imgArr = ReadTensorFromImageFile(img_buffer_);
            ConfigProto config = new ConfigProto();
            GPUOptions gpuConfig = new GPUOptions();
            gpuConfig.AllowGrowth = true;
            gpuConfig.PerProcessGpuMemoryFraction = 0.3;
            config.GpuOptions = gpuConfig;

            using (var sess = tf.Session(detectorGraph, config))
            {
                Tensor tensorClasses = detectorGraph.OperationByName("Identity");
                Tensor imgTensor = detectorGraph.OperationByName("x");
                Tensor[] outTensorArr = new Tensor[] { tensorClasses };

                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

                //Console.WriteLine($"Results: {results[0].ToString()}");
                return PreProcessEmotion(img_buffer_, results[0]);
            }
        }

        private NDArray PreProcessEmotion(NDArray img_buffer_, NDArray bboxes)
        {
            var results = bboxes.GetNDArrays();
            if (results.Length > 0)
            {
                var bbox = results[0];
                var coor = bbox[new Slice(stop: 4)].astype(NPTypeCode.Float);
                var (x1, y1) = ((int)(coor[0].GetValue<float>() * width), (int)(coor[1].GetValue<float>() * height));
                var (x2, y2) = ((int)(coor[2].GetValue<float>() * width), (int)(coor[3].GetValue<float>() * height));
                Console.WriteLine($"x1, y1 : {(int)x1}, {(int)y1}");
                Console.WriteLine($"x2, y2 : {(int)x2}, {(int)y2}");

                var crop_img = img_buffer_[new Slice(y1, y2 + 1), new Slice(x1, x2 + 1), Slice.All];
                NDArray img_buffer_1 = new NDArray(crop_img.ToArray<byte>());
                img_buffer_1 = img_buffer_1.reshape((y2 - y1 + 1, x2 - x1 + 1, 3));
                return img_buffer_1;
                //InferenceEmotion(crop_img);
                //crop_img = crop_img[Slice.All, Slice.Ellipsis ::- 1, Slice.All]
            }
            return null;
        }

        private void InferenceEmotion(int shotsTaken, NDArray img_buffer_)
        {
            var imgArr = ReadTensorFromDetected(img_buffer_, img_size: 60);
            ConfigProto config = new ConfigProto();
            GPUOptions gpuConfig = new GPUOptions();
            gpuConfig.AllowGrowth = true;
            gpuConfig.PerProcessGpuMemoryFraction = 0.3;
            config.GpuOptions = gpuConfig;

            using (var sess = tf.Session(emotionGraph, config))
            {
                Tensor tensorClasses = emotionGraph.OperationByName("Identity");
                Tensor imgTensor = emotionGraph.OperationByName("x");
                Tensor[] outTensorArr = new Tensor[] { tensorClasses };

                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

                var emotions = results[0].ToArray<float>();
                //var records = new List<object>
                //{
                //    new { Frame = shotsTaken, Results = results[0] },
                //};
                //csv.WriteRecord(new { Frame = shotsTaken, Results = results[0] });
                //csv.Flush();
                var record = new CSVRecord();
                record.Neutral = (int)(Math.Round(emotions[0], 2) * 100);
                record.Happy = (int)(Math.Round(emotions[1], 2) * 100);
                record.Sad = (int)(Math.Round(emotions[2], 2) * 100);
                record.Angry = (int)(Math.Round(emotions[3], 2) * 100);
                record.Surprised = (int)(Math.Round(emotions[4], 2) * 100);
                record.Date = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

                using (var stream = File.Open("output.csv", FileMode.Append))
                using (var writer = new StreamWriter(stream))
                using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
                {
                    // Don't write the header again.
                    csv.Configuration.HasHeaderRecord = false;
                    csv.WriteRecord<CSVRecord>(record);
                    csv.NextRecord();
                }

                Console.WriteLine($"Results: {results[0].ToString()}");
                //PreProcessEmotion(img_buffer, results[0]);
            }
        }

        private void buildOutputImage(NDArray img_buffer_, NDArray bboxes)
        {
            // var rnd = new Random();
            //var classes = File.ReadAllLines(@"D:\SciSharp\SciSharp-Stack-Examples\data\classes\coco.names");
            //var classes = ["person"];
            //var num_classes = len(classes);
            //var (image_h, image_w) = (image.shape[0], image.shape[1]);
            // var hsv_tuples = range(num_classes).Select(x => (rnd.Next(255), rnd.Next(255), rnd.Next(255))).ToArray
            var results = bboxes;
            var imageDir = "Outputs";

            foreach (var (i, bbox) in enumerate(results.GetNDArrays()))
            {
                var coor = bbox[new Slice(stop: 4)].astype(NPTypeCode.Float);
                var (x1, y1) = ((int)(coor[0].GetValue<float>() * width), (int)(coor[1].GetValue<float>() * height));
                var (x2, y2) = ((int)(coor[2].GetValue<float>() * width), (int)(coor[3].GetValue<float>() * height));
                Console.WriteLine($"x1, y1 : {(int)x1}, {(int)y1}");
                Console.WriteLine($"x2, y2 : {(int)x2}, {(int)y2}");

                var fontScale = 0.5;
                float score = bbox[15];
                var class_ind = (float)bbox[5];
                var bbox_color = (0, 255, 0);// hsv_tuples[rnd.Next(num_classes)];
                var bbox_thick = (int)(0.6 * (width + height) / 600);

                //var cropped_image = image[(y1, y2), (x1, x2)];
                //cv2.imwrite(Path.Combine(imageDir, "Output_" + i + ".jpg"), cropped_image);
                //cv2.imshow("Detected Objects in TensorFlow.NET", cropped_image);
                //cv2.waitKey();

                //cv2.rectangle(image, ((int)x1, (int)y1), ((int)x2, (int)y2), bbox_color, 2);

                // show label;
                //var bbox_mess = "aa";//$"{classes[(int)class_ind]}: {score.ToString("P")}";
                //var t_size = cv2.getTextSize(bbox_mess, HersheyFonts.HERSHEY_SIMPLEX, fontScale, thickness: bbox_thick / 2);
                //cv2.rectangle(image, (coor[0], coor[1]), (coor[0] + t_size.Width, coor[1] - t_size.Height - 3), bbox_color, -1);
                //cv2.putText(image, bbox_mess, (coor[0], coor[1] - 2), HersheyFonts.HERSHEY_SIMPLEX,
                //        fontScale, (0, 0, 0), bbox_thick / 2, lineType: LineTypes.LINE_AA);
                //break;

                //var cropped_image = image[(y1, y2), (x1, x2)];




                //cv2.imcrop

            }
            //cv2.rectangle(image, (369, 382), (402, 414), (0, 255, 0), 2);

            //return image;
        }

        private void InferenceEmotion(Graph graph, NDArray imgArr)
        {
            using (var sess = tf.Session(graph))
            {
                Tensor tensorClasses = graph.OperationByName("Identity");
                Tensor imgTensor = graph.OperationByName("x");
                Tensor[] outTensorArr = new Tensor[] { tensorClasses };

                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

                Console.WriteLine($"Results: {results[0].ToString()}");
                //buildOutputImage(original_image, results);
            }
        }

        private NDArray ReadTensorFromImageFile(NDArray img_buffer_, int img_size = 640)
        {
            var graph = tf.Graph().as_default();
            ConfigProto config = new ConfigProto();
            GPUOptions gpuConfig = new GPUOptions();
            gpuConfig.AllowGrowth = true;
            gpuConfig.PerProcessGpuMemoryFraction = 0.3;
            config.GpuOptions = gpuConfig;

            var t3 = tf.constant(img_buffer_, dtype: TF_DataType.TF_UINT8);
            //var inp = tf.reshape(t3, (height, width, 3));
            var casted = tf.cast(t3, tf.float32);
            var dims_expander = tf.expand_dims(casted, 0);
            var resize = tf.constant(new int[] { img_size, img_size });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            using (var sess = tf.Session(graph, config))
                return sess.run(bilinear);   
        }

        private NDArray ReadTensorFromDetected(NDArray img_buffer_, int img_size = 60)
        {
            var graph = tf.Graph().as_default();
            ConfigProto config = new ConfigProto();
            GPUOptions gpuConfig = new GPUOptions();
            gpuConfig.AllowGrowth = true;
            gpuConfig.PerProcessGpuMemoryFraction = 0.3;
            config.GpuOptions = gpuConfig;

            var t3 = tf.constant(img_buffer_, dtype: TF_DataType.TF_UINT8);
            //var inp = tf.reshape(t3, (height, width, 3));
            var casted = tf.cast(t3, tf.float32);
            var dims_expander = tf.expand_dims(casted, 0);
            var resize = tf.constant(new int[] { img_size, img_size });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            using (var sess = tf.Session(graph, config))
                return sess.run(bilinear);
        }

        private void Screenshot(byte[] SreenBuffer)
        {
            using (var BMP = new Bitmap(width, height))
            {
                using (var g = Graphics.FromImage(BMP))
                {
                    g.CopyFromScreen(new Point(x, y), Point.Empty, new Size(width, height), CopyPixelOperation.SourceCopy);

                    if (captureCursor)
                    {
                        CURSORINFO pci;
                        pci.cbSize = Marshal.SizeOf(typeof(CURSORINFO));

                        if (GetCursorInfo(out pci))
                        {
                            if (pci.flags == CURSOR_SHOWING)
                            {
                                DrawIcon(g.GetHdc(), pci.ptScreenPos.x, pci.ptScreenPos.y, pci.hCursor);
                                g.ReleaseHdc();
                            }
                        }
                    }

                    g.Flush();

                    //var nd = BMP.ToNDArray(flat: false, copy: false, discardAlpha: true);

                    var bits = BMP.LockBits(
                        new Rectangle(0, 0, width, height),
                        ImageLockMode.ReadOnly,
                        PixelFormat.Format24bppRgb
                    );
                    Marshal.Copy(bits.Scan0, SreenBuffer, 0, SreenBuffer.Length);

                    //var nd = new NDArray(NPTypeCode.Byte, Shape.Vector(bits.Stride * BMP.Height), fillZeros: false);
                    //unsafe
                    //{ 
                    //    // Get the respective addresses
                    //    byte* src = (byte*)bits.Scan0;
                    //    byte* dst = (byte*)nd.Unsafe.Address; //we can use unsafe because we just allocated that array and we know for sure it is contagious.
                    //
                    //    // Copy the RGB values into the array.
                    //    System.Buffer.MemoryCopy(src, dst, nd.size, nd.size); //faster than Marshal.Copy                    
                    //}

                    //var nd1 = nd.reshape(height, width, 3).flat;
                    //var nd2 = nd1.reshape(height, width);

                    //Console.WriteLine($"nd shape {nd.Shape}");
                    //Console.WriteLine($"nd shape {nd1.Shape}");

                    //cv2.imshow("Detected Objects in TensorFlow.NET", nd1);
                    //cv2.waitKey();
                    //var mat = new sharpCV.Mat(nd);
                    //var src = sharpCV.Mat(2, 2, CV_8UC3, data);
                    //cv2.imshow("Detected Objects in TensorFlow.NET", mat);
                    //cv2.waitKey();

                    //var nd = BMP.ToNDArray(flat: false, copy: true, discardAlpha: true);
                    BMP.UnlockBits(bits);
                }
            }
        }

        private void audioSource_DataAvailable(object sender, WaveInEventArgs e)
        {
            int vol = 0;

            if (waveFile != null)
            {
                if (Mute)
                {
                    waveFile.Write(new byte[e.BytesRecorded], 0, e.BytesRecorded);
                }
                else
                {
                    waveFile.Write(e.Buffer, 0, e.BytesRecorded);
                }
                waveFile.Flush();
            }

            var signalled = WaitHandle.WaitAny(new WaitHandle[] { videoFrameWritten, stopThread });
            if (signalled == 0)
            {
                if (audioSource.WaveFormat.BitsPerSample == 32)
                {
                    if (Mute)
                    {
                        audioStream.WriteBlock(new byte[e.BytesRecorded / 2], 0, e.BytesRecorded / 2);
                    }
                    else
                    {
                        byte[] newArray16Bit = new byte[e.BytesRecorded / 2];
                        short two;
                        float value;
                        for (int i = 0, j = 0; i < e.BytesRecorded; i += 4, j += 2)
                        {
                            value = (BitConverter.ToSingle(e.Buffer, i));
                            two = (short)(value * short.MaxValue);

                            newArray16Bit[j] = (byte)(two & 0xFF);
                            newArray16Bit[j + 1] = (byte)((two >> 8) & 0xFF);
                        }

                        audioStream.WriteBlock(newArray16Bit, 0, e.BytesRecorded / 2);

                        float max = 0;
                        for (int index = 0; index < e.BytesRecorded / 2; index += 2)
                        {
                            short sample = (short)((newArray16Bit[index + 1] << 8) | newArray16Bit[index + 0]);
                            var sample32 = sample / 32768f;
                            if (sample32 < 0) sample32 = -sample32;
                            if (sample32 > max) max = sample32;
                        }

                        vol = (int)(100 * max);
                    }
                }
                else
                {
                    if (Mute)
                    {
                        audioStream.WriteBlock(new byte[e.BytesRecorded], 0, e.BytesRecorded);
                    }
                    else
                    {
                        audioStream.WriteBlock(e.Buffer, 0, e.BytesRecorded);

                        float max = 0;
                        for (int index = 0; index < e.BytesRecorded; index += 2)
                        {
                            short sample = (short)((e.Buffer[index + 1] << 8) | e.Buffer[index + 0]);
                            var sample32 = sample / 32768f;
                            if (sample32 < 0) sample32 = -sample32;
                            if (sample32 > max) max = sample32;
                        }

                        vol = (int)(100 * max);
                    }
                }
                audioBlockWritten.Set();
            }

            OnPeakVolumeChangedArgs opvcArgs = new OnPeakVolumeChangedArgs()
            {
                Volume = vol
            };
            PeakVolumeChanged(opvcArgs);
        }

        protected virtual void PeakVolumeChanged(OnPeakVolumeChangedArgs e)
        {
            EventHandler<OnPeakVolumeChangedArgs> handler = OnPeakVolumeChanged;
            if (handler != null)
            {
                handler(this, e);
            }
        }

        public event EventHandler<OnPeakVolumeChangedArgs> OnPeakVolumeChanged;
    }

    public class OnPeakVolumeChangedArgs : EventArgs
    {
        public int Volume { get; set; }
    }
}
