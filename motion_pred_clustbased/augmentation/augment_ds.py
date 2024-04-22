import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
import numpy as np
import os
import warnings


class DataAugmenter:
    def __init__(self, dl , aug_cfg, gen, clusterer, scgan_trainer) -> None:
        self.type = aug_cfg['type'] # augmentation type from {'balanced', 'synthetic', 'mixed'}
        self.mult = aug_cfg['mult'] # multiplier int 
        self.original_dl = dl
        self.gen = gen
        self.gen.eval()
        self.clusterer = clusterer
        self.ds = scgan_trainer.ds
        self.cs = scgan_trainer.cs
        self.hyp_cfg = scgan_trainer.hyp_cfg
        self.save_plt = scgan_trainer.save_cfg['plots']
        self.path_save = scgan_trainer.save_cfg['path']
        self.obs_len = scgan_trainer.data_cfg['obs_len']
        self.device = gen.device


    def __minmax_vel_checker(self, traj : torch.Tensor):
        mask = torch.logical_and(torch.logical_and(traj[:, :, 0] >= self.vx_min, traj[:, :, 0] <= self.vx_max),  \
                torch.logical_and(traj[:, :, 1] >= self.vy_min, traj[:, :, 1] <= self.vy_max))
        mask = torch.all(mask, dim = 1)
        valid_traj = traj[mask, ...]
        return valid_traj
    
        
    def get_balanced_trajs(self, init_dist : torch.Tensor) -> torch.Tensor:
        nsamples_dom_cluster, cls_dom_cluster = torch.max(init_dist, dim = 0) # most dominant cluster
        print("\nOrginal {} samples on dominant cluster {}".format(nsamples_dom_cluster.item(), cls_dom_cluster.item()))
        new_dist = init_dist.clone() * self.mult
        new_nsamples_dom_cluster = torch.max(new_dist, dim = 0)[0]
        print("After {}x multiplier: orginal {} samples on dominant cluster {}\n".format(self.mult, new_nsamples_dom_cluster.item(), cls_dom_cluster.item()))
        
        nsamples_to_gen = new_nsamples_dom_cluster - init_dist
        return self.generator_n_trajs(nsamples_to_gen)
        
            
    def generator_n_trajs(self, n_trajs : torch.Tensor) -> torch.Tensor:
        print("Samples to generate: {}".format(n_trajs))
        all_labels = torch.from_numpy(self.clusterer.get_label_distribution()[1])
        
        if self.type in ["synthetic", "mixed"]:
            rep_all_labels = all_labels.repeat(self.mult)
            lbl_batches = torch.split(rep_all_labels, self.hyp_cfg['bs'])
        elif self.type == "balanced":
            cl_lb = torch.arange(self.clusterer.k)
            lbl_batches = [cl_lb[i].repeat(cnt) for i, cnt in enumerate(n_trajs)]
            lbl_batches = torch.cat(lbl_batches)    
            idx = torch.randperm(lbl_batches.size(0))
            lbl_batches = lbl_batches[idx]
            lbl_batches = torch.split(lbl_batches, self.hyp_cfg['bs'])

        n_trajs_total, n_curr_trajs = n_trajs.sum(), 0
        
        trajs = []
        with torch.no_grad():
            while n_curr_trajs < n_trajs_total:
                for batch in lbl_batches:
                    dummy_inp = torch.empty_like(batch).to(self.device)
                    pred_traj = self.gen(dummy_inp, labels = batch.to(self.device))
                    valid_trajs = self.__minmax_vel_checker(pred_traj)
                    if len(valid_trajs) > 0:
                        valid_trajs_denormed = self.cs.denorm_coords(valid_trajs.clone())
                        trajs.append(valid_trajs_denormed if n_trajs_total - n_curr_trajs >= self.hyp_cfg['bs'] else valid_trajs_denormed[: n_trajs_total - n_curr_trajs])
                    n_curr_trajs += len(valid_trajs_denormed)
        trajs = torch.cat(trajs, dim = 0) 
        return trajs 
    
    
    def plot_displacements(self, preds, gts):
        y_hat, labels = preds.clone(), None
        if self.clusterer:
            labels = self.clusterer.get_labels(self.cs.norm_coords(y_hat, mode = "vect_dir"), None)
            n_clusters = self.clusterer.k
            if n_clusters > 20:
                warnings.warn("Color visualization limited to 20 clusters")
            cmap = get_cmap("tab20", n_clusters + 1 if n_clusters <=20 else 20) if n_clusters >= 10 else get_cmap("tab10", n_clusters + 1)
            captions = [f"Cluster {i}" for i in range(n_clusters)]
        else:
            cmap =  get_cmap("tab10", 2)
            captions = ["FT-GAN"]
            
        color_list = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        gts_denorm = self.cs.denorm_coords(gts.clone())
        cnt_trajs = np.zeros(len(captions))
        
        print("\nPlotting displacements distribution...")
        fig = plt.figure(figsize=(15,15))
        plt.title("Train set input features alignment", fontsize = 29)
        
        for i in range(len(preds)):
            y_pred = preds[i].cpu().numpy()
            lbl = labels[i] if self.clusterer else 0
            if i < len(gts_denorm):
                gt = gts_denorm[i].cpu().numpy()
                plt.scatter(gt[:, 0], gt[:, 1], s = 5, c = color_list[-1], label = "Ground-Truth Displacements" if i == 0 else "") # gt
                
            plt.scatter(y_pred[:, 0], y_pred[:, 1], s = 5, c = color_list[lbl], label = f"Generated Displacements for {captions[lbl]}" if cnt_trajs[lbl] == 0 else "") # pred
            cnt_trajs[lbl] += 1
        
        x_min, x_max, y_min, y_max = min(gts_denorm[:, :, 0].min(), preds[:, :, 0].min()), \
                                     max(gts_denorm[:, :, 0].max(), preds[:, :, 0].max()), \
                                     min(gts_denorm[:, :, 1].min(), preds[:, :, 1].min()), \
                                     max(gts_denorm[:, :, 1].max(), preds[:, :, 1].max())
          
        plt.legend(prop={'size': 19},  markerscale = 10);
        plt.xlim(x_min - 0.1, x_max + 0.1); 
        plt.ylim(y_min - 0.1, y_max + 0.1); 
        # plt.axis('off');
        plt.tight_layout();
        print("saving displacements plot")
        ps = os.path.join(self.path_save, "plots")
        if not os.path.exists(ps):
            os.makedirs(ps)
        fig.savefig(os.path.join(ps, f"{self.ds}_gen_dispvsgt_disp.svg"), bbox_inches="tight")     
        plt.close(fig)
        return labels, gts_denorm.clone()
        
        
    def plot_synt_ds(self, trajs, labels, gts_denorm):
        if self.clusterer:
            n_clusters = self.clusterer.k
            if n_clusters > 20:
                warnings.warn("Color visualization limited to 20 clusters")
            cmap = get_cmap("tab20", n_clusters + 1 if n_clusters <=19 else 20) if n_clusters > 9 else get_cmap("tab10", n_clusters + 1)
            captions = [f"Cluster {i}" for i in range(n_clusters)]
        else:
            cmap =  get_cmap("tab10", 2)
            captions = ["FT-GAN"]
        cnt_trajs = np.zeros(len(captions))
        color_list = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        gts = torch.cumsum(gts_denorm.clone(), dim = 1)
        trajs_xy = torch.cumsum(trajs.clone(), dim = 1)
        fig = plt.figure(figsize=(15,15))
        plt.title("Train set trajectories", fontsize = 29)
        for i, traj in enumerate(trajs_xy):
            if i < len(gts_denorm):
                gt = gts[i].cpu().numpy()
                plt.plot(gt[:, 0], gt[:, 1], c = color_list[-1], label = "Ground-Truth trajectories" if i == 0 else "")
            lbl = labels[i] if self.clusterer else 0
            plt.plot(traj[:, 0], traj[:, 1], c = color_list[lbl], label = f"Generated trajectories for {captions[lbl]}" if cnt_trajs[lbl] == 0 else "")
            cnt_trajs[lbl] += 1
        plt.legend(prop={'size': 19});
        # plt.axis('off');
        plt.tight_layout();
        print("saving trajectories plot")
        ps = os.path.join(self.path_save, "plots")
        if not os.path.exists(ps):
            os.makedirs(ps)
        fig.savefig(os.path.join(ps, f"{self.ds}_trajtectories_training_set.svg"), bbox_inches="tight")     
        plt.close(fig)
        

        
    def __load_init_settings(self):
        self.vx_min, self.vx_max, self.vy_min, self.vy_max = self.clusterer.x[:, :, 0].min(), \
                                                             self.clusterer.x[:, :, 0].max(), \
                                                             self.clusterer.x[:, :, 1].min(), \
                                                             self.clusterer.x[:, :, 1].max()
        print(f"\n\nRunning {self.type} augmentation with {self.mult}x multiplier.... \n\n")
        labels_dist = torch.tensor(self.clusterer.get_label_distribution()[0]).unsqueeze(dim = -1)
        print("Initial labels distribution:", labels_dist)
        return labels_dist
    
    
    def run(self):
        if self.clusterer:
            init_dist = self.__load_init_settings()
            if self.type == "balanced":
                trajs =  self.get_balanced_trajs(init_dist)
            elif self.type in ["synthetic", "mixed"]:
                trajs =  self.generator_n_trajs(init_dist.clone() * self.mult)
            gts = self.clusterer.x.clone()
        else: # vanilla
            trajs_2_gen = self.original_dl.size(0) * self.mult
            print("Trajectories to generate from FT-GAN:", trajs_2_gen)
            self.vx_min, self.vx_max, self.vy_min, self.vy_max = self.original_dl[:, :, 0].min(), \
                                                                 self.original_dl[:, :, 0].max(), \
                                                                 self.original_dl[:, :, 1].min(), \
                                                                 self.original_dl[:, :, 1].max(),
            trajs, n = [], 0
            with torch.no_grad():
                while n < trajs_2_gen: 
                    dummy_inp = torch.empty(self.hyp_cfg['bs']).to(self.device)
                    pred_traj = self.gen(dummy_inp)
                    valid_trajs = self.__minmax_vel_checker(pred_traj)
                    if len(valid_trajs) > 0:
                        valid_trajs_denormed = self.cs.denorm_coords(valid_trajs.clone())
                        trajs.append(valid_trajs_denormed if trajs_2_gen - n >= self.hyp_cfg['bs'] else valid_trajs_denormed[: trajs_2_gen - n])
                    n += len(valid_trajs)
            trajs = torch.cat(trajs, dim = 0) 
            gts = self.original_dl.clone()
        if self.save_plt:
            labels, gts_denorm = self.plot_displacements(trajs, gts)
            self.plot_synt_ds(trajs, labels, gts_denorm)
            
        return AugmentDs(trajs, self.ds, self.obs_len)

class AugmentDs(Dataset):
    def __init__(self, disps, ds, obs_len) -> None:
        super().__init__()
        self.samples = disps
        self.ds = ds
        self.obs_len = obs_len
        
        
    def __getitem__(self, idx):
        # TODO: add random noise?
        disps = self.samples[idx]
        zeros = torch.zeros((1, disps.size(-1))) 
        xy_traj = torch.cat([zeros, disps.clone()], dim = 0)
        xy = torch.cumsum(xy_traj, dim = 0)
        r_polar = torch.sqrt(disps[:, 0]**2 + disps[:, 1]**2)
        t_polar = torch.arctan2(disps[:, 1], disps[:, 0])
        polar = torch.stack([r_polar, t_polar], -1)
        x, y, dx, dy, x_polar, y_polar = xy[:self.obs_len], xy[self.obs_len:], \
                                         disps[:self.obs_len - 1], disps[self.obs_len - 1:], \
                                         polar[:self.obs_len - 1], polar[self.obs_len - 1:]
        
        return (x, y, dx, dy, x_polar, y_polar, torch.empty(self.obs_len)) if self.ds in ["thor", "argoverse"] else (x, y, dx, dy, x_polar, y_polar)
        
    def __len__(self):
        return len(self.samples)
    
    

#TODOS:
# 2) add random noise in our synthetic samples?
# 3) Change path saving plots, etc.