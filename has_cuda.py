import torch

if __name__ == "__main__":
  has_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  print(f"has_cuda = {has_cuda}")
  print(f"device = {device}")
  print(f"n_gpu = {n_gpu}")



