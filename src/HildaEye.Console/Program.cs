using OpenCvSharp;
using OpenCvSharp.Face;


Console.WriteLine("HILDA - EYE");

Console.WriteLine("loading training images");
var pictures = new List<Mat>();
var humanIndexes = new List<int>();
var firstnames = new List<string>
{
    "ulysse",
    "math"
};

for (var i = 0; i <= 1; i++)
{
    var folderPath = $"training/human_{i}";
    var files = Directory.GetFiles(folderPath);

    Console.WriteLine($"Loading face pictures for : {firstnames[i]}");
    var totalPictures = files.Length;
    var currentPictureIndex = 0;
    foreach (var file in files)
    {
        currentPictureIndex++;
        Console.WriteLine($"{currentPictureIndex} / {totalPictures}");

        var image = Cv2.ImRead(file);
        var gray = new Mat();
        Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        
        pictures.Add(gray);
        humanIndexes.Add(i);
    }
}

Console.WriteLine("Loading face pictures finished...");
Console.WriteLine();
Console.WriteLine("Open your eyes");
Console.WriteLine();

try
{
    var recognizer = LBPHFaceRecognizer.Create(1, 8, 8, 8, 100);
    recognizer.Train(pictures, humanIndexes);

    using var capture = new VideoCapture(0);
    if (!capture.IsOpened())
    {
        Console.WriteLine("UNREACHABLE_CAMERA");
        return;
    }

    var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

    while (true)
    {
        using (var frame = capture.RetrieveMat())
        {
            var gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
                
            var faces = faceCascade.DetectMultiScale(gray, 1.1, 3, HaarDetectionTypes.DoCannyPruning, new Size(30, 30));
                
            foreach (var face in faces)
            {
                var detectedFace = gray.SubMat(face);
                    
                int index;
                double confidence;
                recognizer.Predict(detectedFace, out index, out confidence);

                var text = string.Empty;
                if (index > -1)
                {
                    if (index < firstnames.Count)
                    {
                        var name = $"{firstnames[index]}";
                        text = $"{name} ({confidence})";
                    }
                    else
                    {
                        text = $"human_{index} ({confidence})";
                    }
                }
                else
                {
                    text = "UNKNOWN";
                }

                Cv2.Rectangle(frame, face, Scalar.LimeGreen, 2);
                Cv2.PutText(frame, text, new Point(face.X, face.Y - 10), HersheyFonts.HersheyPlain, 1, Scalar.Red);
            }
                
            Cv2.ImShow("HILDA_EYE_ANALYZER", frame);
        }

        var k = Cv2.WaitKey(1);
        if (k == 27)
        {
            break;
        }
    }
}
catch (Exception e)
{

    Console.WriteLine($"something went wrong : {e}");
}



Console.WriteLine("end of app, press anything to leave");