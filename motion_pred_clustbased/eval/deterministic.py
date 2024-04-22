import torch
from .common import Evaluator
from tqdm import tqdm
import os


class DetEval(Evaluator):
    """ LSTM evaluator
    """
    def __init__(self, dl, cfg, inputs_type, outputs_type, device, writer=None):
        super().__init__(dl, cfg, inputs_type, outputs_type, device, writer)
        self.path_save = os.path.join(self.save_dir, "plots")
    
    def evaluate(self, model, epoch):
        model.eval()
        with tqdm(self.dl, unit = "batch") as tval:
            with torch.no_grad():
                obss, outs, preds = [], [], []
                for batch in tval:
                    tval.set_description("Eval Det predictor")
                    if self.ds == "benchmark" :
                        obs, pred, obs_vec_dir, pred_vect_dir, obs_polar, pred_polar = batch
                    else: # argoverse or thor
                        obs, pred, obs_vec_dir, pred_vect_dir, obs_polar, pred_polar, raw_labels = batch
                    obs, pred, obs_vec_dir, pred_vect_dir, obs_polar, pred_polar = obs.to(self.device), pred.to(self.device), obs_vec_dir.to(self.device), pred_vect_dir.to(self.device), obs_polar.to(self.device), pred_polar.to(self.device)
                    obs_norm, pred_norm, obs_vectdir_norm, pred_vectdir_norm, obs_polar_norm, pred_polar_norm = self.cs.norm_input(x = obs, y = pred, x_vect_dir = obs_vec_dir, y_vect_dir = pred_vect_dir, x_polar = obs_polar, y_polar = pred_polar)
                
                
                    if self.inputs_type == "dx, dy":
                        inp_obs, inp_pred = obs_vectdir_norm, pred_vectdir_norm
                    elif self.inputs_type == "px, py":
                        inp_obs, inp_pred = obs_polar_norm, pred_polar_norm
                    elif self.inputs_type == "dx, dy, px, py":
                        inp_obs, inp_pred = torch.cat([obs_vectdir_norm, obs_polar_norm], -1), torch.cat([pred_vectdir_norm, pred_polar_norm], -1)
                    else:
                        inp_obs, inp_pred = (obs_vectdir_norm, pred_vectdir_norm) if self.outputs_type == "vect_dir" else (obs_polar_norm, pred_polar_norm)

                    out = model(inp_obs, coord_scaler = self.cs)
                    out = out.view(out.shape[0], self.test_seq_len, -1) # (bs, sl, 2)
                    out = self.cs.denorm_increment(obs, out)
                    self.update_metrics(out, pred)
                    obss.append(obs), outs.append(out), preds.append(pred)
                if  self.save_plot_trajs:
                        self.save_plot(obss, outs, preds, epoch)
                self.compute_metrics()
                self.logging_metrics(epoch)
                self.reset_metrics()
                if self.writer is not None:
                    self.write_metrics(epoch)