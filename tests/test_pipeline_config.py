import yaml
with open("config/pipeline_config.yaml", "r") as f:
    config = yaml.safe_load(f)
print("Config loaded:", config["project"]["id"])