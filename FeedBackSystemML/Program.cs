using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace FeedBackSystemML
{
    class Program
    {
        static List<FeedBackTrainingData> trainingData = new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData = new List<FeedBackTrainingData>();

        static void LoadTestData()
        {
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is good",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "very horrible",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "so nice",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "just bland",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "very disturbing",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "beautiful thing",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "i love this",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "admirable",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "uninteresting",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "game changer",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "very unbearable",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "adorable",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "unappreciative",
                IsGood = false

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "sweet and nice",
                IsGood = true

            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "thank you",
                IsGood = true

            });

        }
        static void LoadTrainingData()
        {
            trainingData.Add(new FeedBackTrainingData() { 
                FeedBackTest = "so good",
                IsGood = true
                
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is horrible",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is great",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "very poor",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "sweet and nice",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "exquisite",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is outstanding",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is rubbish",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "admirable",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "cankanterous",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "extremely beautiful",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "unbearable",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "so bland",
                IsGood = false

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "this is the game changer",
                IsGood = true

            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackTest = "i love this",
                IsGood = true

            });

        }
        static void Main(string[] args)
        {
            // Load training Data
            LoadTrainingData();

            // create an instance of ML Context
            var mlContext = new MLContext();

            // Convert your data to IDataView
            IDataView dataView = mlContext.CreateStreamingDataView<FeedBackTrainingData>(trainingData);

            // create the pipeline and define workflows
            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackTest", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

            // Train algorithm and generate model 
            var model = pipeline.Fit(dataView);

            // ---------------------- Testing the data -----------

            // Load the test data to check for accuracy
            LoadTestData();
            IDataView dataView1 = mlContext.CreateStreamingDataView<FeedBackTrainingData>(testData);

            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);



            // Use the model
            string cont = "Y";
            while (cont == "Y")
            {
                Console.WriteLine("Enter Feedback string");
                string feebackString = Console.ReadLine().ToString();

                var predictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPredictionData>(mlContext);
                var feedbackInput = new FeedBackTrainingData();
                feedbackInput.FeedBackTest = feebackString;
                var feedbackPredicted = predictionFunction.Predict(feedbackInput);
                Console.WriteLine("Predicted :- " + feedbackPredicted.IsGood);
                Console.ReadLine();

            }

        }
    }
}
