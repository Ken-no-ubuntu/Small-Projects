import torch

# Compute gradients of the model with respect to the input
def compute_gradients(model, inputs, labels):
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
    
    # Backpropagate to compute gradients
    loss.backward()
    
    # Get the gradients of the inputs
    gradients = inputs.grad.abs().mean(dim=0).cpu().numpy()
    return gradients

# Integrated Gradients method
def integrated_gradients(model, inputs, labels, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    # Scale inputs and compute gradients at each step
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    total_gradients = torch.zeros_like(inputs)
    
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        output = model(scaled_input)
        loss = torch.nn.functional.binary_cross_entropy(output, labels)
        loss.backward()
        total_gradients += scaled_input.grad
    
    # Average gradients across steps and multiply by the input difference
    avg_gradients = total_gradients / steps
    integrated_grads = (inputs - baseline) * avg_gradients
    return integrated_grads.abs().mean(dim=0).cpu().numpy()
