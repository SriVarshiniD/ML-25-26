


class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        """
        weights: list or array of input weights
        threshold: neuron firing threshold
        """
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        """
        inputs: list or array of input signals (0 or 1)
        returns: output 0 or 1
        """
        total_input = sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if total_input >= self.threshold else 0
# AND gate: output 1 only if both inputs are 1
and_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=2)

print("AND Gate")
for x1 in [0,1]:
    for x2 in [0,1]:
        y = and_neuron.activate([x1, x2])
        print(f"Input: {x1},{x2} -> Output: {y}")
# OR gate: output 1 if at least one input is 1
or_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)

print("\nOR Gate")
for x1 in [0,1]:
    for x2 in [0,1]:
        y = or_neuron.activate([x1, x2])
        print(f"Input: {x1},{x2} -> Output: {y}")
# NOT gate: single input, threshold = 0.5
not_neuron = McCullochPittsNeuron(weights=[-1], threshold=-0.5)

print("\nNOT Gate")
for x in [0,1]:
    y = not_neuron.activate([x])
    print(f"Input: {x} -> Output: {y}")
