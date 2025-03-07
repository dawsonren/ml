import config
import time
from cross_validation import cross_validate

def compare():
    current_hyperparams = config.CONFIG

    best_params = None
    for new_config in config.HYPERPARAM_LIST:
        for param in new_config:
            current_hyperparams[param] = new_config[param]
        
        print(current_hyperparams)
        start_timer = time.time()
        accuracy = cross_validate(current_hyperparams, False)
        print(f"model accuracy: {accuracy:.2f}%, time taken: {time.time() - start_timer}")

        if best_params == None or best_params[1] < accuracy:
            best_params = [current_hyperparams, accuracy]
    
    print(f"best found hyperparams: {best_params}")
    return best_params
