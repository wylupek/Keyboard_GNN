# Define the function to calculate FAR and FRR
def calculate_far_frr(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize a list to store the results
    results = []
    
    # Process each line
    for line in lines:
        # Parse the data from the line
        parts = line.strip().split()
        model = parts[0].split(':')[1]
        tp = int(parts[1].split(':')[1])
        fp = int(parts[2].split(':')[1])
        tn = int(parts[3].split(':')[1])
        fn = int(parts[4].split(':')[1])
        
        # Calculate FAR and FRR
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # Append the results
        results.append({
            'Model': model,
            'FAR': far,
            'FRR': frr
        })
    
    # Print the results
    print("Model\tFAR\t\tFRR")
    for result in results:
        print(f"{result['Model']} & {result['FAR']:.3f} & {result['FRR']:.3f} \\\\")
        print(f"\\hline")

    # Calculate the average FAR and FRR
    total_far = sum(result['FAR'] for result in results)
    total_frr = sum(result['FRR'] for result in results)
    average_far = total_far / len(results)
    average_frr = total_frr / len(results)

    # Print the average FAR and FRR
    print(f"Average FAR: {average_far:.3f}")
    print(f"Average FRR: {average_frr:.3f}")


# Path to the input text file
import sys
file_path = sys.argv[1]

# Call the function
calculate_far_frr(file_path)

