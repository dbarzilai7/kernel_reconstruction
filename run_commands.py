import numpy as np

DEFAULT_CMD =  "python reconstruction.py --recovery_size 500 --train_size 500 --init_std 0.3 --lr 2e-2 --coefs_lr 1e-2 --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --use_test_time_seed "
DEFAULT_CMD_n100 =  "python reconstruction.py --recovery_size 100 --train_size 100 --init_std 0.3 --lr 2e-2 --coefs_lr 1e-2 --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --use_test_time_seed "


commands = []

# NTK 500 images CIFAR10
commands += ["python reconstruction.py --recovery_size 500 --train_size 500 --normalize_to_sphere --init_std 0.1 --alpha_init_std 0.5 --lr 1e-4 --coefs_lr 1e-3 --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --use_test_time_seed --recovery_model ntk_analytic --recovery_layers 3 --target_model ntk_analytic --target_layers 3 --dataset CIFAR5M --batch_size 500000 --num_steps 1000000"]

# Poly kernel 500 images CIFAR10
commands += ["python reconstruction.py --recovery_size 500 --train_size 500 --init_std 0.3 --lr 5e-3 --coefs_lr 5e-3 --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --use_test_time_seed --recovery_model poly_kernel --recovery_gamma 0.001 --target_model poly_kernel --target_gamma 0.001 --dataset CIFAR5M --batch_size 500000 --num_steps 300000 --target_task krr --high_freq_loss 1e-5"]

# # VAE
commands += [DEFAULT_CMD + f"--recovery_model laplace --recovery_gamma 0.03 --target_model laplace --target_gamma 0.03 --dataset celebA --batch_size 50000 --recovery_size 400 --train_size 400 --num_steps 150000 --init_std 0.01 --scheduler onecycle --max_img_size 128 --vae_path taesd3 --checkpoint_interval 1000 --checkpoint_dir ckpts/CIFAR10/job_array_test"]

# # # Laplace n=100 various equations
num_samples = list(range(5000, 50001, 5000))
pca_dims  = [0, 768, 1536]
commands += [DEFAULT_CMD_n100 + f"--recovery_model laplace --recovery_gamma 0.15 --target_model laplace --target_gamma 0.15 --dataset CIFAR10 --batch_size {int(samples)} --num_steps 150000 --recovery_pca_dim {pca_dim}" for samples in num_samples for pca_dim in pca_dims]

# # Eq/param runs n=500 CIFAR5M
params_over_output_dim = int(((32 * 32 * 3) + 10) * 500 / 10)
num_samples = np.arange(0.2, 2.1, 0.2) * params_over_output_dim
pca_dims = [0]
gammas = [0.15]
kernels = ['laplace']
target_tasks = ['krr']
commands += [DEFAULT_CMD + f"--recovery_model {kernels[i]} --recovery_gamma {gammas[i]} --target_model {kernels[i]} --target_gamma {gammas[i]} --dataset CIFAR5M --batch_size {int(samples)} --num_steps 300000 --target_task {taks} --recovery_pca_dim {pca_dim}" for samples in num_samples for pca_dim in pca_dims for i in range(len(kernels)) for taks in target_tasks]


# Laplace gamma experiment Assunming gamma = 0.15 run separately
gammas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
commands += [DEFAULT_CMD + f"--recovery_model laplace --recovery_gamma {gamma} --target_model laplace --target_gamma {gamma} --dataset CIFAR5M --batch_size 500000 --num_steps 300000 --target_task krr" for gamma in gammas]

#Laplace SVM CIFAR10
commands += [DEFAULT_CMD + "--recovery_model laplace --recovery_gamma 0.15 --target_model laplace --target_gamma 0.15 --dataset CIFAR5M --batch_size 500000 --num_steps 300000 --target_task svm"]

# RBF SVM CIFAR10
commands += [DEFAULT_CMD + "--recovery_model gaussian --recovery_gamma 0.003 --target_model gaussian --target_gamma 0.003 --dataset CIFAR5M --batch_size 500000 --num_steps 300000 --target_task svm"]

# RBF KRR CIFAR10
commands += [DEFAULT_CMD + "--recovery_model gaussian --recovery_gamma 0.003 --target_model gaussian --target_gamma 0.003 --dataset CIFAR5M --batch_size 500000 --num_steps 300000 --target_task krr"]

# Laplace SVM CIFAR100
commands += [DEFAULT_CMD + "--recovery_model laplace --recovery_gamma 0.15 --target_model laplace --target_gamma 0.15 --dataset CIFAR100 --batch_size 50000 --vert_flip --horiz_flip --vert_horz_flip --num_steps 300000 --target_task svm"]

# Laplace KRR CIFAR100
commands += [DEFAULT_CMD + "--recovery_model laplace --recovery_gamma 0.15 --target_model laplace --target_gamma 0.15 --dataset CIFAR100 --batch_size 50000 --vert_flip --horiz_flip --vert_horz_flip --num_steps 300000"]

# RBF SVM CIFAR100
commands += [DEFAULT_CMD + "--recovery_model gaussian --recovery_gamma 0.003 --target_model gaussian --target_gamma 0.003 --dataset CIFAR100 --batch_size 50000 --vert_flip --horiz_flip --vert_horz_flip --num_steps 300000 --target_task svm"]

# RBF KRR CIFAR100
commands += [DEFAULT_CMD + "--recovery_model gaussian --recovery_gamma 0.003 --target_model gaussian --target_gamma 0.003 --dataset CIFAR100 --batch_size 50000 --vert_flip --horiz_flip --vert_horz_flip --num_steps 300000"]

# RBF KRR celebA
commands += [DEFAULT_CMD + "--recovery_model gaussian --recovery_gamma 0.0005 --target_model gaussian --target_gamma 0.0005 --dataset celebA --max_img_size 64 --batch_size 200000 --num_steps 150000 --target_task krr"]

# Laplace KRR celebA
commands += [DEFAULT_CMD + "--recovery_model laplace --recovery_gamma 0.03 --target_model laplace --target_gamma 0.03 --dataset celebA --max_img_size 64 --batch_size 200000 --num_steps 150000 --target_task krr"]

# RBF celebA VAE
commands += ["python reconstruction.py --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --recovery_model gaussian --recovery_gamma 0.0001 --target_model gaussian --target_gamma 0.0001 --dataset celebA --batch_size 60000 --recovery_size 500 --train_size 500 --num_steps 150000 --init_std 0.01 --max_img_size 128 --vae_path taesd3 --lr 2e-2 --coefs_lr 1e-2 --use_test_time_seed"]

# # Laplace celebA VAE
commands += ["python reconstruction.py --scheduler onecycle --plot_all_recons --save_metrics --output_dir outputs --save_recons --recovery_model laplace --recovery_gamma 0.03 --target_model laplace --target_gamma 0.03 --dataset celebA --batch_size 60000 --recovery_size 500 --train_size 500 --num_steps 150000 --init_std 0.01 --max_img_size 128 --vae_path taesd3 --lr 2e-2 --coefs_lr 1e-2 --use_test_time_seed"]