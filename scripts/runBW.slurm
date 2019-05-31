tarch
NAME="Red Hat Enterprise Linux Server"
VERSION="7.6 (Maipo)"
ID="rhel"
ID_LIKE="fedora"
VARIANT="Server"
VARIANT_ID="server"
VERSION_ID="7.6"
PRETTY_NAME="Red Hat Enterprise Linux"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:redhat:enterprise_linux:7.6:GA:server"
HOME_URL="https://www.redhat.com/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"

REDHAT_BUGZILLA_PRODUCT="Red Hat Enterprise Linux 7"
REDHAT_BUGZILLA_PRODUCT_VERSION=7.6
REDHAT_SUPPORT_PRODUCT="Red Hat Enterprise Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="7.6"
Red Hat Enterprise Linux Server release 7.6 (Maipo)
/fslhome/tarch/anaconda3/envs/cycleGAN/bin/python
----------------- Options ---------------
               batch_size: 1                             
                    beta1: 0.5                           
          checkpoints_dir: ./checkpoints                 
           continue_train: False                         
                 dataroot: /fslhome/tarch/compute/research/handwriting/MUNIT/datasets/handwriting	[default: None]
             dataset_mode: unaligned                     
                direction: AtoB                          
              display_env: main                          
             display_freq: 500                           	[default: 400]
               display_id: 0                             	[default: 1]
            display_ncols: 4                             
             display_port: 8097                          
           display_server: http://localhost              
          display_winsize: 256                           
                    epoch: latest                        
              epoch_count: 1                             
                 fineSize: 256                           
                fineSizeX: 1280                          	[default: 256]
                fineSizeY: 64                            	[default: 256]
      generate_on_the_fly: False                         
                  gpu_ids: 0                             
                init_gain: 0.02                          
                init_type: normal                        
                 input_nc: 1                             	[default: 3]
                  isTrain: True                          	[default: None]
                 lambda_A: 10.0                          
                 lambda_B: 10.0                          
          lambda_identity: 0.5                           
                 loadSize: 286                           
                loadSizeX: 1280                          	[default: 286]
                loadSizeY: 64                            	[default: 286]
                       lr: 0.0002                        
           lr_decay_iters: 50                            
                lr_policy: lambda                        
         max_dataset_size: inf                           
                    model: cycle_gan                     
               n_layers_D: 3                             
                     name: handwriting_cyclegan_BW       	[default: experiment_name]
                      ndf: 64                            
                     netD: basic                         
                     netG: resnet_9blocks                
                      ngf: 64                            
                    niter: 100                           
              niter_decay: 100                           
               no_dropout: True                          
                  no_flip: True                          	[default: False]
                  no_html: False                         
                 no_lsgan: False                         
                     norm: instance                      
              num_threads: 4                             
                output_nc: 1                             	[default: 3]
                    phase: train                         
                pool_size: 50                            
               print_freq: 100                           
           resize_or_crop: resize_and_crop               
          save_epoch_freq: 5                             
         save_latest_freq: 500                           	[default: 5000]
           serial_batches: False                         
                     step: None                          
                   suffix:                               
         update_html_freq: 500                           	[default: 1000]
                  verbose: False                         
----------------- End -------------------
dataset [UnalignedDataset] was created
#training images = 13078
initialize network with normal
initialize network with normal
initialize network with normal
initialize network with normal
model [CycleGANModel] was created
---------- Networks initialized -------------
[Network G_A] Total number of parameters : 11.366 M
[Network G_B] Total number of parameters : 11.366 M
[Network D_A] Total number of parameters : 2.763 M
[Network D_B] Total number of parameters : 2.763 M
-----------------------------------------------
create web directory ./checkpoints/handwriting_cyclegan_BW/web...
(epoch: 1, iters: 100, time: 1.499, data: 0.374) D_A: 0.345 G_A: 0.486 cycle_A: 0.261 idt_A: 0.216 D_B: 0.567 G_B: 0.307 cycle_B: 0.484 idt_B: 0.117 
(epoch: 1, iters: 200, time: 1.503, data: 0.001) D_A: 0.245 G_A: 0.265 cycle_A: 0.248 idt_A: 0.242 D_B: 0.293 G_B: 0.365 cycle_B: 0.689 idt_B: 0.102 
(epoch: 1, iters: 300, time: 1.507, data: 0.001) D_A: 0.363 G_A: 0.129 cycle_A: 0.173 idt_A: 0.551 D_B: 0.246 G_B: 0.308 cycle_B: 1.202 idt_B: 0.095 
(epoch: 1, iters: 400, time: 1.497, data: 0.001) D_A: 0.418 G_A: 0.383 cycle_A: 0.181 idt_A: 0.202 D_B: 0.263 G_B: 0.299 cycle_B: 0.670 idt_B: 0.086 
(epoch: 1, iters: 500, time: 1.701, data: 0.001) D_A: 0.228 G_A: 0.272 cycle_A: 0.162 idt_A: 0.227 D_B: 0.216 G_B: 0.531 cycle_B: 0.506 idt_B: 0.077 
saving the latest model (epoch 1, total_steps 500)
(epoch: 1, iters: 600, time: 1.502, data: 0.001) D_A: 0.233 G_A: 0.209 cycle_A: 0.248 idt_A: 0.225 D_B: 0.317 G_B: 0.503 cycle_B: 0.386 idt_B: 0.136 
(epoch: 1, iters: 700, time: 1.500, data: 0.001) D_A: 0.291 G_A: 0.293 cycle_A: 0.241 idt_A: 0.249 D_B: 0.224 G_B: 0.369 cycle_B: 0.440 idt_B: 0.114 
(epoch: 1, iters: 800, time: 1.499, data: 0.001) D_A: 0.315 G_A: 0.578 cycle_A: 0.156 idt_A: 0.227 D_B: 0.186 G_B: 0.211 cycle_B: 0.570 idt_B: 0.077 
(epoch: 1, iters: 900, time: 1.497, data: 0.001) D_A: 0.273 G_A: 0.243 cycle_A: 0.176 idt_A: 0.130 D_B: 0.236 G_B: 0.187 cycle_B: 0.310 idt_B: 0.082 
(epoch: 1, iters: 1000, time: 1.662, data: 0.001) D_A: 0.216 G_A: 0.240 cycle_A: 0.167 idt_A: 0.098 D_B: 0.176 G_B: 0.293 cycle_B: 0.318 idt_B: 0.082 
saving the latest model (epoch 1, total_steps 1000)
(epoch: 1, iters: 1100, time: 1.497, data: 0.001) D_A: 0.159 G_A: 0.303 cycle_A: 0.153 idt_A: 0.366 D_B: 0.183 G_B: 0.864 cycle_B: 1.065 idt_B: 0.080 
(epoch: 1, iters: 1200, time: 1.496, data: 0.001) D_A: 0.306 G_A: 0.266 cycle_A: 0.251 idt_A: 0.161 D_B: 0.194 G_B: 0.645 cycle_B: 0.364 idt_B: 0.087 
(epoch: 1, iters: 1300, time: 1.493, data: 0.001) D_A: 0.305 G_A: 0.305 cycle_A: 0.138 idt_A: 0.167 D_B: 0.301 G_B: 0.104 cycle_B: 0.396 idt_B: 0.068 
(epoch: 1, iters: 1400, time: 1.502, data: 0.001) D_A: 0.321 G_A: 0.512 cycle_A: 0.231 idt_A: 0.122 D_B: 0.207 G_B: 0.196 cycle_B: 0.416 idt_B: 0.082 
(epoch: 1, iters: 1500, time: 1.649, data: 0.001) D_A: 0.274 G_A: 0.396 cycle_A: 0.136 idt_A: 0.125 D_B: 0.257 G_B: 0.165 cycle_B: 0.288 idt_B: 0.062 
saving the latest model (epoch 1, total_steps 1500)
(epoch: 1, iters: 1600, time: 1.502, data: 0.001) D_A: 0.299 G_A: 0.330 cycle_A: 0.161 idt_A: 0.125 D_B: 0.207 G_B: 0.522 cycle_B: 0.386 idt_B: 0.068 
(epoch: 1, iters: 1700, time: 1.500, data: 0.001) D_A: 0.195 G_A: 0.287 cycle_A: 0.174 idt_A: 0.089 D_B: 0.152 G_B: 0.300 cycle_B: 0.250 idt_B: 0.068 
(epoch: 1, iters: 1800, time: 1.498, data: 0.001) D_A: 0.248 G_A: 0.381 cycle_A: 0.197 idt_A: 0.151 D_B: 0.268 G_B: 0.279 cycle_B: 0.400 idt_B: 0.104 
(epoch: 1, iters: 1900, time: 1.493, data: 0.001) D_A: 0.242 G_A: 0.217 cycle_A: 0.141 idt_A: 0.180 D_B: 0.177 G_B: 0.310 cycle_B: 0.501 idt_B: 0.067 
(epoch: 1, iters: 2000, time: 1.643, data: 0.001) D_A: 0.232 G_A: 0.319 cycle_A: 0.250 idt_A: 0.151 D_B: 0.182 G_B: 0.310 cycle_B: 0.337 idt_B: 0.059 
saving the latest model (epoch 1, total_steps 2000)
(epoch: 1, iters: 2100, time: 1.496, data: 0.001) D_A: 0.240 G_A: 0.307 cycle_A: 0.179 idt_A: 0.097 D_B: 0.289 G_B: 0.176 cycle_B: 0.241 idt_B: 0.069 
(epoch: 1, iters: 2200, time: 1.492, data: 0.001) D_A: 0.215 G_A: 0.247 cycle_A: 0.142 idt_A: 0.152 D_B: 0.211 G_B: 0.231 cycle_B: 0.455 idt_B: 0.063 
(epoch: 1, iters: 2300, time: 1.499, data: 0.001) D_A: 0.177 G_A: 0.337 cycle_A: 0.111 idt_A: 0.109 D_B: 0.248 G_B: 0.317 cycle_B: 0.232 idt_B: 0.046 
(epoch: 1, iters: 2400, time: 1.492, data: 0.001) D_A: 0.257 G_A: 0.312 cycle_A: 0.125 idt_A: 0.103 D_B: 0.273 G_B: 0.121 cycle_B: 0.338 idt_B: 0.059 
(epoch: 1, iters: 2500, time: 1.654, data: 0.001) D_A: 0.243 G_A: 0.344 cycle_A: 0.177 idt_A: 0.180 D_B: 0.130 G_B: 0.099 cycle_B: 0.531 idt_B: 0.080 
saving the latest model (epoch 1, total_steps 2500)
(epoch: 1, iters: 2600, time: 1.497, data: 0.001) D_A: 0.181 G_A: 0.378 cycle_A: 0.150 idt_A: 0.130 D_B: 0.201 G_B: 0.315 cycle_B: 0.363 idt_B: 0.060 
(epoch: 1, iters: 2700, time: 1.497, data: 0.001) D_A: 0.321 G_A: 0.270 cycle_A: 0.133 idt_A: 0.208 D_B: 0.343 G_B: 0.260 cycle_B: 0.533 idt_B: 0.055 
(epoch: 1, iters: 2800, time: 1.497, data: 0.001) D_A: 0.219 G_A: 0.268 cycle_A: 0.151 idt_A: 0.094 D_B: 0.164 G_B: 0.378 cycle_B: 0.305 idt_B: 0.058 
(epoch: 1, iters: 2900, time: 1.494, data: 0.001) D_A: 0.210 G_A: 0.358 cycle_A: 0.121 idt_A: 0.134 D_B: 0.160 G_B: 0.364 cycle_B: 0.352 idt_B: 0.053 
(epoch: 1, iters: 3000, time: 1.653, data: 0.001) D_A: 0.231 G_A: 0.276 cycle_A: 0.209 idt_A: 0.103 D_B: 0.362 G_B: 0.500 cycle_B: 0.398 idt_B: 0.100 
saving the latest model (epoch 1, total_steps 3000)
(epoch: 1, iters: 3100, time: 1.504, data: 0.001) D_A: 0.339 G_A: 0.209 cycle_A: 0.307 idt_A: 0.145 D_B: 0.451 G_B: 0.242 cycle_B: 0.329 idt_B: 0.171 
(epoch: 1, iters: 3200, time: 1.498, data: 0.001) D_A: 0.304 G_A: 0.172 cycle_A: 0.173 idt_A: 0.093 D_B: 0.226 G_B: 0.479 cycle_B: 0.343 idt_B: 0.066 
(epoch: 1, iters: 3300, time: 1.497, data: 0.001) D_A: 0.205 G_A: 0.373 cycle_A: 0.153 idt_A: 0.079 D_B: 0.387 G_B: 0.104 cycle_B: 0.194 idt_B: 0.068 
(epoch: 1, iters: 3400, time: 1.503, data: 0.001) D_A: 0.177 G_A: 0.219 cycle_A: 0.188 idt_A: 0.232 D_B: 0.258 G_B: 0.456 cycle_B: 0.519 idt_B: 0.060 
(epoch: 1, iters: 3500, time: 1.651, data: 0.001) D_A: 0.296 G_A: 0.220 cycle_A: 0.135 idt_A: 0.095 D_B: 0.197 G_B: 0.151 cycle_B: 0.282 idt_B: 0.066 
saving the latest model (epoch 1, total_steps 3500)
(epoch: 1, iters: 3600, time: 1.504, data: 0.001) D_A: 0.199 G_A: 0.367 cycle_A: 0.183 idt_A: 0.199 D_B: 0.153 G_B: 0.244 cycle_B: 0.479 idt_B: 0.045 
(epoch: 1, iters: 3700, time: 1.502, data: 0.001) D_A: 0.241 G_A: 0.317 cycle_A: 0.325 idt_A: 0.090 D_B: 0.346 G_B: 0.620 cycle_B: 0.318 idt_B: 0.146 
(epoch: 1, iters: 3800, time: 1.499, data: 0.001) D_A: 0.260 G_A: 0.365 cycle_A: 0.477 idt_A: 0.191 D_B: 0.182 G_B: 0.420 cycle_B: 0.658 idt_B: 0.204 
(epoch: 1, iters: 3900, time: 1.508, data: 0.001) D_A: 0.085 G_A: 0.510 cycle_A: 0.195 idt_A: 0.082 D_B: 0.107 G_B: 0.259 cycle_B: 0.342 idt_B: 0.072 
(epoch: 1, iters: 4000, time: 1.657, data: 0.001) D_A: 0.207 G_A: 0.162 cycle_A: 0.578 idt_A: 0.159 D_B: 0.741 G_B: 0.477 cycle_B: 0.326 idt_B: 0.272 
saving the latest model (epoch 1, total_steps 4000)
(epoch: 1, iters: 4100, time: 1.501, data: 0.001) D_A: 0.213 G_A: 0.121 cycle_A: 0.371 idt_A: 0.098 D_B: 0.373 G_B: 0.243 cycle_B: 0.197 idt_B: 0.137 
(epoch: 1, iters: 4200, time: 1.500, data: 0.001) D_A: 0.207 G_A: 0.353 cycle_A: 0.183 idt_A: 0.121 D_B: 0.219 G_B: 0.199 cycle_B: 0.336 idt_B: 0.056 
(epoch: 1, iters: 4300, time: 1.511, data: 0.001) D_A: 0.276 G_A: 0.183 cycle_A: 0.165 idt_A: 0.140 D_B: 0.198 G_B: 0.208 cycle_B: 0.397 idt_B: 0.061 
(epoch: 1, iters: 4400, time: 1.499, data: 0.001) D_A: 0.182 G_A: 0.474 cycle_A: 0.180 idt_A: 0.193 D_B: 0.255 G_B: 0.700 cycle_B: 0.525 idt_B: 0.060 
(epoch: 1, iters: 4500, time: 1.647, data: 0.001) D_A: 0.156 G_A: 0.536 cycle_A: 0.142 idt_A: 0.232 D_B: 0.221 G_B: 0.324 cycle_B: 0.585 idt_B: 0.061 
saving the latest model (epoch 1, total_steps 4500)
(epoch: 1, iters: 4600, time: 1.499, data: 0.001) D_A: 0.161 G_A: 0.365 cycle_A: 0.161 idt_A: 0.140 D_B: 0.148 G_B: 0.391 cycle_B: 0.433 idt_B: 0.082 
(epoch: 1, iters: 4700, time: 1.499, data: 0.001) D_A: 0.230 G_A: 0.258 cycle_A: 0.194 idt_A: 0.169 D_B: 0.206 G_B: 0.598 cycle_B: 0.438 idt_B: 0.080 
(epoch: 1, iters: 4800, time: 1.498, data: 0.001) D_A: 0.273 G_A: 0.285 cycle_A: 0.339 idt_A: 0.133 D_B: 0.301 G_B: 0.334 cycle_B: 0.288 idt_B: 0.148 
slurmstepd: error: *** JOB 30861056 ON m8g-1-3 CANCELLED AT 2019-05-30T15:32:48 DUE TO TIME LIMIT ***
