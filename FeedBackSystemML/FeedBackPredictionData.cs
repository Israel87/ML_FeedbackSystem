using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace FeedBackSystemML
{
    public class FeedBackPredictionData
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }
}
