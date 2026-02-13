
CLI Args
  ├─ ModelParams (stage, paths, save settings)
  ├─ OptimizationParams (epochs, loss weights, log/save frequency)
  └─ NetworkParams (net config)
        │
        ▼
prepare_output_and_logger()
  ├─ create output/<run_id>/
  ├─ create output/<run_id>/log/
  └─ (optional) TensorBoard SummaryWriter(output/<run_id>/)
        │
        ▼
AvatarModel(model, net, opt, train=True)
  ├─ net (main renderer / predictor)
  ├─ pose, transl (trainable in stage1; frozen in stage2)
  ├─ pose_encoder (trained in stage2)
  ├─ optimizers/schedulers (training_setup)
  └─ dataloader (getTrainDataloader)
        │
        ▼
(Optional) Resume
  ├─ load checkpoint → set epoch_start, first_iter
  └─ if stage2: stage_load(stage1_out_path)
        │
        ▼
========================== TRAIN LOOP ==========================
For epoch = epoch_start+1 ... epochs:
  ├─ Set train/eval modes
  │    stage1: net, pose, transl → train
  │    stage2: net, pose_encoder → train ; pose, transl → eval/frozen
  │
  ├─ wdecay_rgl = decay(lambda_rgl, epoch, every=20)
  │
  └─ For each batch in train_loader:
        │
        ├─ batch_data → GPU (to_cuda)
        ├─ gt_image = batch_data["original_image"]
        │
        ├─ Forward
        │    stage1: image, points, offset_loss, geo_loss, scale_loss
        │    stage2: image, points, pose_loss,  offset_loss
        │
        ├─ Losses
        │    Ll1        = (1-λdssim) * L1(image, gt)
        │    Lssim      =  λdssim    * (1 - SSIM(image, gt))
        │    Loffset    =  wdecay_rgl * offset_loss
        │
        │    stage1 total:
        │      L = λscale*scale_loss + Loffset + Ll1 + Lssim + geo_loss
        │
        │    stage2 total:
        │      L = Loffset + Ll1 + Lssim + 10*pose_loss
        │
        ├─ (optional after lpips_start_iter)
        │    Llpips = λlpips * LPIPS(image, gt)
        │    L += Llpips
        │
        ├─ Backprop + step
        │    zero_grad → backward → optimizer.step
        │
        ├─ Periodic artifacts (every log_iter)
        │    - save pred points as PLY (trimesh)
        │    - save pred/gt images as PNG
        │
        └─ TensorBoard scalars
             - total loss, l1, offset, (geo or pose), lpips, iter_time

End epoch:
  └─ periodic checkpoint save (every model.save_epoch, after save_epochs[0])

