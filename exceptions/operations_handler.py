import logging, pathlib


current_working_directory = pathlib.Path.cwd()
print(current_working_directory)
def setup_logger(logger_name: str, log_file: str, log_level = logging.INFO) -> logging.Logger:
    """
    this function allows the system to create and write log data of the system's operations,
    Args:
        logger_name (str): name of the log file to create
        log_file (str): the file path to the log file
        log_level (int): the value of the long type (warn, info, debug)
    """
    # create a log from a specific logger name
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)
    return logger


def create_folder_and_log_file(folder_name:str, file_name:str) -> pathlib.Path:
    """
    this function creates a folder and a corresponding log file
    in that folder
    Args:
        folder_name (str): name of the log file to create
        log_file (str): use file Path to the log file
        log_level (int)
    """
    new_path = current_working_directory.joinpath(folder_name)
    # create the folder path only once by checking if it has already been created
    new_path.mkdir(exist_ok=True)
    log_file_path = new_path.joinpath(file_name)
    # create the file if it does not exist
    log_file_path.touch()

folder_name = 'logs'
log_files_to_create = ['system.log', 'userops.log', 'llmresponse.log', 'evals.log']
for log_file in log_files_to_create:
    create_folder_and_log_file(folder_name=folder_name, file_name=log_file)


system_logger = setup_logger(__name__, f'{current_working_directory}/logs/system.log') # system logger
userops_logger = setup_logger("UserLogger", f'{current_working_directory}/logs/userops.log') # userops logger
llmresponse_logger = setup_logger("LLMResponse", f'{current_working_directory}/logs/llmresponse.log') # llmresponse logger
evalresponse_logger = setup_logger("EvalsReponse", f'{current_working_directory}/logs/evals.log', 1) # evaluation logger