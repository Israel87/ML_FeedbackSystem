using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace FeedBackSystemML
{
    public class FeedBackTrainingData
    {
        // Labels are output expected from the algorithm
        [Column(ordinal:"0", name: "Label")]
        public bool IsGood { get; set; }
        // Features are Input sent to the algorithm
        [Column(ordinal: "1")]
        public string FeedBackTest { get; set; }
 
    }
}
