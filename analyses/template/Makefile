#GLOBALS
USER=admornhja
PROJECT=/media/data/bids-analysis-platform/analyses/template
MAKE_DATA=src/data/make_dataset.py
MAKE_REPORT=src/reports/make_report.py

PYTHON_ENV=/media/data/Environments/Orn/base_python/bin/activate 
REMOTE_HOST=cha-datasc-01.nix.csmodule.com
HOST=$(shell hostname)

host:
ifeq ($(HOST),$(REMOTE_HOST))
	@echo 'on server'
	$(error on server)
	REMOTE=false
else
	@echo 'on local'
	REMOTE=true
endif

file-location:
	@echo $(shell pwd)

local-clean-reports: clean-fig clean-ppt
	@echo "cleaning done"
local-clean-fig:
	rm -rf reports/figures/*
local-clean-ppt:
	rm -rf reports/powerpoint/*
local-clean-data:
	rm -rf data/raw/*

#RUN REMOTE
python_env='source $(PYTHON_ENV);'
python_env+='python --version;'
proj_dir='cd $(PROJECT);'
make_dataset='python $(MAKE_DATA);'
make_report='python $(MAKE_REPORT);'
remote-run: host
	ssh -t $(USER)@$(REMOTE_HOST) $(python_env)$(proj_dir)$(make_dataset)$(make_report)
remote-make-data: host
	ssh -t $(USER)@$(REMOTE_HOST) $(python_env)$(proj_dir)$(make_dataset)
remote-make-report: host
	ssh -t $(USER)@$(REMOTE_HOST) $(python_env)$(proj_dir)$(make_report)

#GET
figures=$(PROJECT)'/reports/figures/*'
reports=$(PROJECT)'/reports/*'
data=$(PROJECT)'/data/raw/*'
ppt=$(PROJECT)'/reports/powerpoint/*'
get-reports: host local-clean-reports
	scp -r $(USER)@$(REMOTE_HOST):$(reports) reports/
get-data: host local-clean-data
	scp -r $(USER)@$(REMOTE_HOST):$(data) data/raw/
get-fig: host local-clean-fig
	scp -r $(USER)@$(REMOTE_HOST):$(figures) reports/figures/
get-ppt: host local-clean-ppt
	scp -r $(USER)@$(REMOTE_HOST):$(ppt) reports/powerpoint/

#GIT
git_status='git status;'
git_pull='git pull;'
connect: host
	ssh -t $(USER)@$(REMOTE_HOST) $(proj_dir)bash
remote-git-status: host
	ssh -t $(USER)@$(REMOTE_HOST) $(proj_dir)$(git_status)
remote-git-pull: host
	ssh -t $(USER)@$(REMOTE_HOST) $(proj_dir)$(git_pull)

local-git-acp:
	git add .
	git commit -m 'message'
	git push
