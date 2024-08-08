import argparse
import RobustMetricLib as rm

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluate robustness of trajectories using RobustMetricLib.")
    
    # Add arguments
    parser.add_argument('pose_file1', type=str, help="Path to the first input poses file")
    parser.add_argument('pose_file2', type=str, help="Path to compare with first input poses file")
    parser.add_argument('keep_freq', type=int, help="Argument controlling low pass filter (frequency domain)")
    parser.add_argument('trans_threshold', type=float, help="Translation robustness threshold")
    parser.add_argument('rot_threshold', type=float, help="Rotation robustness threshold")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    rm.evaluate_robustness(args.pose_file1, args.pose_file2, args.keep_freq, args.trans_threshold, args.rot_threshold)

if __name__ == "__main__":
    main()
