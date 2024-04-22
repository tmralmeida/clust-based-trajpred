import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

from ..networks.deterministic import DetNet
from ..eval.deterministic import DetEval
from ..datasets.loaders import *
from ..utils.common import *
from ..utils.deterministic import *


class Trainer():
    def __init__(self,
                 cfg,
                 ) -> None:
        self.cfg = cfg
        self.data_cfg = self.cfg["data"]
        self.input_data = self.data_cfg["inputs"]
        self.hyp_cfg = self.cfg["hyperparameters"]
        self.save_cfg = self.cfg["save"]
        self.model_type = self.cfg["model"]
        self.ds = self.data_cfg["dataset"] # main dataset's name 
        self.test_name = self.data_cfg["dataset_target"] # test dataset
        self.outputs = self.data_cfg["output"]
        self.inputs = ""
        if self.input_data["vect_dir"]:
            self.inputs += "dx, dy"  
        if self.input_data["polar"]:
            str_inp = "px, py" if self.inputs == "" else ", px, py"
            self.inputs += str_inp
        self.inp_dim = len(self.inputs.split(",")) if self.input_data["vect_dir"] or self.input_data["polar"] else 0
        print(f"The input is {str(self.inputs)} == {self.inp_dim}")

        if self.ds == "benchmark":
            print(f"Training {self.model_type} on {self.test_name} from {self.ds}!")
        elif self.ds == "argoverse":
            print(f"Training {self.model_type} on {self.ds}")
        elif self.ds == "thor":
            print(f"Training {self.model_type} on {self.ds}")
        else:
            raise NameError(f"{self.ds} not implemented.")
        
    
    def common_settings(self):
        # Loading common sets
        val_ds = load_ds("val", self.data_cfg)
        
        self.val_dl = DataLoader(val_ds, 
                                 self.hyp_cfg["bs"])
        self.device = torch.device("cpu")

        # Coord scaler
        self.cs = CoordsScaler(self.cfg)
        
    def train(self, *args, **kwargs):
        
        torch.random.seed()
        runs_dir, output_dir = create_dirs(self.cfg, self.inputs) # create output files
        writer = SummaryWriter(runs_dir)
        if "aug_dataset" in kwargs:
            train_dl = kwargs["aug_dataset"]
            print("\n\n=======================================================Training new model on synthetic dataset=======================================================\n\n")
        else:
            train_ds = load_ds("train", self.data_cfg)
            train_dl = DataLoader(train_ds, 
                                self.hyp_cfg["bs"],
                                shuffle = True)
                
        self.common_settings()
        net_cfg = self.cfg["network"]
        net_cfg["model"] = self.model_type
        model = DetNet(cfg = net_cfg,
                       inp_dim = self.inp_dim,
                       device = self.device)

        # Loss
        criterion = det_loss

        # Optimizer
        opt = optim.Adam(model.parameters(), lr = float(self.hyp_cfg["lr"]))
        lr_sched = optim.lr_scheduler.StepLR(opt, int(self.hyp_cfg["step_size"]))


        # Validation and Testing
        validator = DetEval(self.val_dl, 
                            self.cfg, 
                            self.inputs,
                            self.outputs, 
                            self.device,
                            writer = writer)
        
        testing = False
        if self.data_cfg["test"]:
            test_ds = load_ds("test", self.data_cfg)
            testing = True
            test_dl = DataLoader(test_ds, 
                                     self.hyp_cfg["bs"])
            tester = DetEval(test_dl, 
                             self.cfg, 
                             self.inputs,
                             self.outputs, 
                             self.device)
        

        metrics, models = {"ade": [],
                           "fde": []}, []
        # training
        pred_len = self.data_cfg["pred_len"]
        for epoch in range(self.hyp_cfg["max_epochs"]):
            losses = []
            with tqdm(train_dl, unit="batch") as tepoch:
                for batch in tepoch:
                    obs, pred, inp_obs, inp_pred, _, _ = get_batch(batch, self.cs, self.model_type, self.ds, self.inputs, self.device, clustering = None, gen = None, cond_state = None, output_type = self.data_cfg["output"])
                    tepoch.set_description(f"Epoch {epoch + 1}")
                  

                    inp_obs, inp_pred, obs, pred = inp_obs.to(self.device), inp_pred.to(self.device), obs.to(self.device), pred.to(self.device)
                    
                    out = model(inp_obs, coord_scaler = self.cs)
                    opt.zero_grad()
                    loss = criterion(out, obs, pred, pred_len, self.cs).mean()
                    loss.backward()
                    opt.step()
                    
                    losses.append(loss.item())
                    avg_loss = sum(losses)/len(losses)
                    tepoch.set_postfix({"loss" : avg_loss})
                writer.add_scalar("loss_avg", avg_loss, epoch)
                writer.add_scalar("lr", lr_sched.get_last_lr()[-1], epoch)
                lr_sched.step()
                
            #saving checkpoints
            if (epoch % self.save_cfg["checkpoints"]) == 0:
                save_checkpoints_det(epoch, output_dir, self.data_cfg["dataset_target"], self.cfg["model"], model, opt, lr_sched)    
                
            if (epoch % self.hyp_cfg["val_freq"]) == 0:
                models.append(model)
                validator.evaluate(model, epoch + 1)
                ade, fde = validator.ade_res, validator.fde_res
                metrics["ade"].append(ade)
                metrics["fde"].append(fde)
                ep_bm = metrics[self.cfg["save"]["best_metric"]].index(min(metrics[self.cfg["save"]["best_metric"]])) \
                                 if self.cfg["save"]["best_metric"] else -1

                if self.cfg["save"]["best_metric"] and epoch - ep_bm*self.hyp_cfg["val_freq"] >= self.hyp_cfg["patience"]:
                    break
        writer.close()
        test_ep = ep_bm*self.hyp_cfg["val_freq"] if ep_bm != -1 else -1
        if self.cfg["save"]["objects"]:
            save_objects_det(self.model_type,
                            models[ep_bm], 
                            test_ep, 
                            output_dir)
        
        results = {k : v[ep_bm] for k, v in metrics.items()}
        if testing:
            tester.evaluate(models[ep_bm], test_ep)
            results = {"ade" : tester.ade_res, "fde" : tester.fde_res}
            results_line = [f"ade: {tester.ade_res}\n",f"fde: {tester.fde_res} \n"]
        else:
            results_line = [f"{k}: {v[ep_bm]}\n" for k, v in metrics.items()]
        
        if self.cfg["save"]["final_results"]:
            file = open(os.path.join(output_dir,"final_results.txt"),"w")
            file.writelines(results_line)
            file.close()    
        
        return results, model