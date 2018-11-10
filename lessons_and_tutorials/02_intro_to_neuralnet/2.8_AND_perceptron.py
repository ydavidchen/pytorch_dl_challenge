# Section 2.8 AND Perceptron

## Question 1: Implement the AND Perceptron
import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1.0;
weight2 = 1.0;
bias = -2;

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


# Question 2: What are two ways to go from an AND perceptron to an OR perceptron?
# Answer:
## _(blank1)_ (increase or decrease) _(blank2)_ (single weight or the weights)? (Answer: blank1=increase; blank2=weights)
## _(blank3)_ (increase or decrease) magnitude of bias? (Answer: decrease)
