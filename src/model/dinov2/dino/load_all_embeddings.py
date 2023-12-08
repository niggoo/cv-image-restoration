import os
import glob
from safetensors.torch import safe_open


def load_all_tensors(directory_path: str) -> dict:
    """Exemplary way of loading all tensors from a safetensors directory.

    Args:
        directory_path (str): Path to the directory containing the safetensors files.

    Raises:
        RuntimeError: If an error occurs while processing a file. Should never happen!

    Returns:
        dict: Dictionary containing the loaded safetensor dicts. The keys are the file paths.
    """
    # Find all safetensor files
    tensor_paths = glob.glob(
        os.path.join(directory_path, "**/*.safetensors"), recursive=True
    )

    # Dictionary to store loaded tensors
    all_tensors = {}

    for tensor_path in tensor_paths:
        try:
            # Use safe_open to open and read the tensor file
            with safe_open(tensor_path, framework="pt", device="cpu") as f:
                tensors = {}
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

            # Store the loaded tensors from each file in the all_tensors dictionary
            all_tensors[tensor_path] = tensors

        except Exception as e:
            raise RuntimeError(
                f"Error occurred while processing file {tensor_path}: {e}"
            )

    return all_tensors


def main():
    try:
        loaded_tensors = load_all_tensors("../data/embeddings/full/")
        print(f"Loaded tensors from {len(loaded_tensors)} files.")
    except RuntimeError as e:
        print(e)
    print(list(loaded_tensors.keys())[0])
    print(list(loaded_tensors.values())[0]["last_hidden_states"])


if __name__ == "__main__":
    main()
