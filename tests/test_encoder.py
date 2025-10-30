"""
Test encoder with real data samples
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder.encoder import PestDataEncoder
from src.utils import save_json


def test_encoder_with_real_data(csv_path: str, num_samples: int = 3):
    """
    Test encoder with real cleaned data
    
    Args:
        csv_path: Path to cleaned CSV
        num_samples: Number of tehsils to test
    """
    print("="*60)
    print("ENCODER TESTING WITH REAL DATA")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Select diverse samples (low, medium, high severity)
    print(f"\n🎯 Selecting {num_samples} diverse samples...")
    
    # Sort by total pests to get variety
    df_sorted = df.sort_values('TOTAL_PESTS_ABOVE_ETL')
    
    sample_indices = [
        len(df_sorted) // 4,      # Low severity
        len(df_sorted) // 2,      # Medium severity
        len(df_sorted) * 3 // 4   # High severity
    ][:num_samples]
    
    samples = df_sorted.iloc[sample_indices]
    
    # Initialize encoder
    print("\n🤖 Initializing encoder...")
    encoder = PestDataEncoder()
    
    # Test each sample
    results = []
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'='*60}")
        print(f"TEST {idx}/{num_samples}")
        print(f"{'='*60}")
        
        try:
            result = encoder.analyze(row.to_dict(), save_output=True)
            results.append(result)
            
            # Print summary
            print(f"\n📊 SUMMARY:")
            print(f"   Tehsil: {result['tehsil']}")
            print(f"   Risk Level: {result['encoder_analysis']['risk_level']}")
            print(f"   Urgency: {result['encoder_analysis']['action_urgency']}")
            print(f"   Primary Threats: {', '.join(result['encoder_analysis']['primary_threats'])}")
            print(f"   Fallback Used: {'Yes' if result['encoder_analysis'].get('_fallback') else 'No'}")
            
            charts = result.get('charts', {})
            if charts:
                sample_chart = next(iter(charts.values()))
                print(f"   Charts Generated: {len(charts)} (e.g., {sample_chart})")
            
            report_path = result.get('report_path')
            if report_path:
                print(f"   PDF Report: {report_path}")
            
        except Exception as e:
            print(f"❌ Error analyzing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save batch summary
    print(f"\n{'='*60}")
    print("SAVING BATCH SUMMARY")
    print(f"{'='*60}")
    
    summary = {
        'test_date': pd.Timestamp.now().isoformat(),
        'total_samples': len(results),
        'successful': len([r for r in results if not r['encoder_analysis'].get('_fallback')]),
        'fallback': len([r for r in results if r['encoder_analysis'].get('_fallback')]),
        'results': results
    }
    
    summary_path = f"outputs/encoder_results/batch_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(summary, summary_path)
    print(f"💾 Batch summary saved: {summary_path}")
    
    print(f"\n✅ Testing complete!")
    print(f"   Successful: {summary['successful']}/{summary['total_samples']}")
    print(f"   Fallback: {summary['fallback']}/{summary['total_samples']}")
    
    if results and getattr(encoder, "visualizer", None):
        print("\n🗺️  Creating batch visualizations for reviewed tehsils...")
        batch_outputs = encoder.visualizer.generate_batch(results)
        print(f"   Batch charts generated for {len(batch_outputs)} tehsil(s)")
    
    return results


if __name__ == "__main__":
    # Test with your cleaned data
    csv_path = "/mnt/e/pest_advisory_system/data/pest_survey_cleaned.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("Please provide the correct path to your cleaned CSV file")
        sys.exit(1)
    
    # Run tests
    results = test_encoder_with_real_data(csv_path, num_samples=3)
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review saved encoder outputs in: outputs/encoder_results/")
    print("2. Check if JSON parsing is working (fallback count should be 0)")
    print("3. Validate that risk levels match severity")
    print("4. Move to Phase 4: Decoder pipeline")
