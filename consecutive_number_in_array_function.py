import numpy as np

def find_consecutive_number_lengths(array, number):
    consecutive_number_lengths = []
    current_length = 0

    for num in array:
        if num == number:
            current_length += 1
        elif current_length > 0:
            consecutive_number_lengths.append(current_length)
            current_length = 0

    if current_length > 0:
        consecutive_number_lengths.append(current_length)

    return consecutive_number_lengths

# Example usage:
if __name__ == "__main__":
    # Assuming you have a NumPy array filled with 0s and 1s
    # input_array = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0])
    input_array = np.array([0,0,0,0,0,-1,-1,0,0,0,1,0,0,0,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,1,0,0,0])
    number = 0 
    lengths = find_consecutive_number_lengths(input_array, number)
    print("Lengths of consecutive ones:", lengths)
