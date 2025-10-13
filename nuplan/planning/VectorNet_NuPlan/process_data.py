import os
from tqdm import tqdm

from utils import config
from utils.dataset import GraphDataset
from utils.data_processor import DataProcessor
from utils.common_utils import (
    get_scenario_map,
    get_filter_parameters
)

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

if __name__ == '__main__':
    # nuplan arguments
    scenarios_per_type = config.SCENARIOS_PER_TYPE
    total_scenarios = config.TOTAL_SCENARIOS
    shuffle_scenarios = True

    sensor_root = None
    db_files = None

    # create folder for processed data
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    # get scenarios
    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(), 
        subsample_ratio_override=0.5
    )

    print("Building scenario ...")
    builder = NuPlanScenarioBuilder(
        config.DATA_PATH, 
        config.MAP_PATH, 
        sensor_root, 
        db_files, 
        config.MAP_VERSION, 
        scenario_mapping=scenario_mapping
    )

    # scenarios for training
    print("Filtering scenario ...")
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            scenarios_per_type, 
            total_scenarios, 
            shuffle_scenarios
        )
    )

    # enable parallel process
    print("Enabling parallel executor ...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    # get scenarios
    print("Getting scenarios ...")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    
    # Process scenarios
    test_count = round(len(scenarios) * config.TEST_SPLIT)
    train_count = len(scenarios) - test_count
    print("Processing ...")
    print(f"Number of scenarios: {len(scenarios)}, train sets: {train_count}, test sets: {test_count}")

    count = 0
    for scenario in tqdm(scenarios):
        # process data
        data_processor = DataProcessor(scenario)
        if count < train_count:
            data, features = data_processor.process(config.TRAIN_PATH)
        else:
            data, features = data_processor.process(config.TEST_PATH)

        count = count + 1

    GraphDataset(config.TRAIN_PATH)
    GraphDataset(config.TEST_PATH)
    train_data_file = os.listdir(config.TRAIN_PATH)
    test_data_file = os.listdir(config.TEST_PATH)
    total_data_file = train_data_file + test_data_file
    for data_path in total_data_file:
        if data_path.endswith('.pkl'):
            pkl_data_path = os.path.join(config.TRAIN_PATH, data_path)
            if os.path.exists(pkl_data_path):
                os.remove(pkl_data_path)
            else:
                os.remove(os.path.join(config.TEST_PATH, data_path))
