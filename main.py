from fastapi import FastAPI, File, UploadFile
import opensmile
import tempfile
import json

app = FastAPI()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def summarize_opensmile_features(features_df):
    row = features_df.iloc[0]

    summary = {
        "speech_analysis": {
            "pitch": {
                "mean": float(row.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0)),
                "stddev": float(row.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0)),
                "rising_slope_mean": float(row.get("F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope", 0)),
                "falling_slope_mean": float(row.get("F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope", 0)),
            },
            "loudness": {
                "mean": float(row.get("loudness_sma3_amean", 0)),
                "stddev": float(row.get("loudness_sma3_stddevNorm", 0)),
                "peaks_per_sec": float(row.get("loudnessPeaksPerSec", 0)),
            },
            "timing": {
                "voiced_segments_per_sec": float(row.get("VoicedSegmentsPerSec", 0)),
                "mean_voiced_segment_length": float(row.get("MeanVoicedSegmentLengthSec", 0)),
                "mean_unvoiced_segment_length": float(row.get("MeanUnvoicedSegmentLength", 0)),
            },
            "stability": {
                "jitter_mean": float(row.get("jitterLocal_sma3nz_amean", 0)),
                "shimmer_mean": float(row.get("shimmerLocaldB_sma3nz_amean", 0)),
                "hnr_mean": float(row.get("HNRdBACF_sma3nz_amean", 0)),
            },
        }
    }
    return summary


@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract OpenSMILE features
    features = smile.process_file(tmp_path)
    available_features = list(features.columns)

    # Summarize for Llama
    summary = summarize_opensmile_features(features)

    return {
        "summary_for_llm": summary,
        "available_features": available_features
    }
