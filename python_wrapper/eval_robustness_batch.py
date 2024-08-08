import argparse
import RobustMetricLib as rm

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Batch evaluate robustness of trajectories using RobustMetricBatchLib.")
    
    # Add arguments
    parser.add_argument('pose_file1', type=str, help="Path to the first input poses file")
    parser.add_argument('pose_file2', type=str, help="Path to compare with first input poses file")
    parser.add_argument('keep_freq', type=int, help="Argument controlling low pass filter (frequency domain)")
    parser.add_argument('sampling_interval_ns', type=int, help="Sampling interval in nanoseconds")
    parser.add_argument('threshold_start', type=float, help="Start of the threshold range (inclusive)")
    parser.add_argument('threshold_end', type=float, help="End of the threshold range (inclusive)")
    parser.add_argument('threshold_interval', type=float, help="Interval for the threshold range")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    rm.eval_robustness_batch(
        args.pose_file1, 
        args.pose_file2, 
        args.keep_freq, 
        args.sampling_interval_ns, 
        args.threshold_start, 
        args.threshold_end, 
        args.threshold_interval
    )

if __name__ == "__main__":
    main()
