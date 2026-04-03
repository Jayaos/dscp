from torch.utils.data import DataLoader




def run_dscp(experiment_config, model_config, device):

    # load data
    # load ConformalPredictionData
    cp_data = load_data()

    for key, item in cp_data.dataset.items():

        train_dataloader = DataLoader(item["train_dataset"], experiment_config.batch_size, shuffle=True)
        valid_dataloader = DataLoader(item["train_dataset"], experiment_config.batch_size, shuffle=True)
        test_dataloader = DataLoader(item["train_dataset"], experiment_config.batch_size, shuffle=True)

    
        if isinstance(model_config, TransformerModelConfig):
            pass

        elif isinstance(model_config, LSTMModelConfig):
            pass


        print("training model...")
        for i in range(experiment_config.num_repeat):
            model, train_log = train_quantile_prediction_model(model, train_dataloader, valid_dataloader, 
                                                            fcp_config.max_epoch, fcp_config.additional_training_epoch,
                                                            fcp_config.learning_rate, fcp_config.early_stop, device=device)
            
    
        # evaluation



def train_quantile_prediction_model(cfg_flow, train_dataloader, valid_dataloader, 
                                                        fcp_config.max_epoch, fcp_config.additional_training_epoch,
                                                        fcp_config.learning_rate, fcp_config.early_stop, device=device):
    ...
    