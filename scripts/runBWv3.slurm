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
             display_freq: 200                           	[default: 400]
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
         save_latest_freq: 200                           	[default: 5000]
           serial_batches: False                         
                     step: None                          
                   suffix:                               
         update_html_freq: 200                           	[default: 1000]
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
(epoch: 1, iters: 100, time: 1.526, data: 0.176) D_A: 0.365 G_A: 0.388 cycle_A: 0.259 idt_A: 0.245 D_B: 0.377 G_B: 0.477 cycle_B: 0.544 idt_B: 0.113 
Exception ignored in: <bound method _DataLoaderIter.__del__ of <torch.utils.data.dataloader._DataLoaderIter object at 0x7f60eb6bd160>>
Traceback (most recent call last):
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 399, in __del__
    self._shutdown_workers()
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 378, in _shutdown_workers
    self.worker_result_queue.get()
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/queues.py", line 337, in get
    return _ForkingPickler.loads(res)
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/resource_sharer.py", line 87, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/connection.py", line 493, in Client
    answer_challenge(c, authkey)
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/connection.py", line 732, in answer_challenge
    message = connection.recv_bytes(256)         # reject large message
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/connection.py", line 216, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/fslhome/tarch/anaconda3/envs/cycleGAN/lib/python3.6/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError: 
Traceback (most recent call last):
  File "train.py", line 37, in <module>
    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, total_steps) # save out images
  File "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/pytorch-CycleGAN/util/visualizer.py", line 139, in display_current_results
    img_path = '{}_{}.png' % (text, label)
TypeError: not all arguments converted during string formatting
