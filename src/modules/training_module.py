import torch
from pytorch_lightning import LightningModule
from utils.gaussian_utils import render_gaussians
from models.model import GaussianModel
from models.gaussian_renderer import GaussianRenderer
from config.options import Options

class Trainer(LightningModule):
    
    def __init__(self, data):
        super().__init__()
        self.model = GaussianModel()


    def forward(self, batch, batch_idx):
        # predicts gaussians from image
        
        pred = self.model(batch, batch_idx)
        return pred

    def render(self, batch, batch_idx):
        # create ModelOutput Class
        inputs, _, metadata = batch
        input_image = inputs.scene_rgb.unsqueeze(0)
        gaussians = self(batch, batch_idx)
        
        options = Options
        renderer = GaussianRenderer(options)
        
        
        view = inputs.cam_data.view_matrix.unsqueeze(0).unsqueeze(0)
        view_proj = inputs.cam_data.view_proj_matrix.unsqueeze(0).unsqueeze(0)
        pos = inputs.cam_data.position.unsqueeze(0).unsqueeze(0)
        
        print(f"Camera Name: {inputs.cam_data.name}")
        print(f"Camera View: {view}")
        print(f"Camera position: {pos}")
        # gaussians: A tensor of predicted gaussians. Shape (B,N,D)
        # cam_view: A tensor of camera extrinsics. Shape (B,V, 4, 4)
        # cam_voew_proj: A tensor of combined cam_view and projection matrix. Shape (B, V, 4, 4)
        # cam_pos: A tensor of camera positions in world coordinates. Shape (B, V, 3)
        results = renderer.render(
            gaussians=gaussians,
            cam_view=view,
            cam_view_proj=view_proj,
            cam_pos=pos,
            bg_color=torch.randn(3, dtype=torch.float32, device=gaussians.device),
            scale_modifier=0.1)
        

        return results
        
    def training_step(self, batch, batch_idx):
        rendered = self.render(batch, batch_idx)
        final_loss = self.loss_func(
            batch,
            rendered,
            self.losses,
            self.weights
        )
        
        opts = self.optimizers()
        
        if ((batch_idx + 1) % 1 == 0 ) or (#self.opts.trainer.accum_iter == 0) or (
            batch_idx + 1 == len(self.train_data)
        ):
            if isinstance(opts, list):
                for opt in opts:
                    opt.zero_grad()
            else:
                opts.zero_grad()
                
        self.manual_backward(final_loss)
        
        if ((batch_idx + 1) %  1 == 0) or ( # self.opts.trainer.accum_iter == 0) or (
            batch_idx + 1 == len(self.train_data)
        ):
            if isinstance(opts, list):
                for opt in opts:
                    opt.step()
            else:
                opts.step()
                
        self.log("loss", final_loss, sync_dist=True, batch_size=self.batch_size)
        #psnr_val = psnr(rendered["render"], batch["image"][0])
        #self.log("train/psnr", psnr_val, sync_dist=True, batch_size=self.batch_size)
        return {"loss": final_loss}
    
    def loss_func(self, batch, pred, losses_dict, loss_weights):
        
        losses = {}
        
        if "rgb_loss" in losses_dict:
            #loss = l1_loss(pred_image, gt_image, mean=False)
            #losses.rgb_loss = torch.mean(loss)
            ...
        
        final_loss = 0
        for name, loss in losses.items():
            loss_index = losses_dict.index(name)
            weight = loss_weights[loss_index]
            final_loss += weight * loss
        return final_loss
    
    def configure_optimizers(self):
        return self.model.optimizer
    
    def train_dataloader(self):
        return self.train_loader