# project_install_ef2

```bash
rm project_install_ef2 -rf
git clone https://github.com/coefficient49/project_install_ef2.git
cp ./project_install_ef2/in* .
```
for installation

```
source  installing_ef2.sh
```

to activate environment:

```
source init_env.profile
```


to run the pipeline
```
cd ~
rm iter_run_.log
screen -d -m -L -Logfile iter_run_.log bash ~/project_install_ef2/iter_runs.sh
```
