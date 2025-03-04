"""
Default configuration for Monocular Depth Estimation project
"""

# Training
learning_rate = 1e-4
batch_size = 16
num_epochs = 100
save_every = 5
visualize_every = 5
num_workers = 4

# Model
teacher_freeze_encoder = True

# Loss weights
lambda_ssi = 1.0  # Scale-shift invariant loss weight
lambda_teacher = 0.5  # Teacher-student loss weight
lambda_gradient = 0.5  # Gradient matching loss weight

# Loss parameters
ssi_alpha = 0.5  # Scale-shift invariant loss alpha parameter
ssi_scales = 4  # Number of scales for multi-scale SSI loss
gradient_scales = 4  # Number of scales for gradient matching loss

# Paths
data_dir = './data'
checkpoint_dir = './checkpoints'
log_dir = './logs'

# Export config as a dictionary
config = {
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'save_every': save_every,
    'visualize_every': visualize_every,
    'num_workers': num_workers,
    'teacher_freeze_encoder': teacher_freeze_encoder,
    'lambda_ssi': lambda_ssi,
    'lambda_teacher': lambda_teacher,
    'lambda_gradient': lambda_gradient,
    'ssi_alpha': ssi_alpha,
    'ssi_scales': ssi_scales,
    'gradient_scales': gradient_scales,
    'data_dir': data_dir,
    'checkpoint_dir': checkpoint_dir,
    'log_dir': log_dir
}
