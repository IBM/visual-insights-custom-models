# IBM Visual Insights - Custom Models


This repository contains examples of custom models for
[IBM Visual Insights](https://www.ibm.com/fr-fr/marketplace/ibm-visual-insights/resources) (previously IBM PowerAI Vision).

It provides samples, explanations on how to use it, as well as some test files allowing to test models locally.

There are currently 2 examples, [__image classification__](./image_classification/) and [__object detection__](./object_detection/). Both have been implemented in Keras and tested in IBM Visual Insights 1.2.0 (and works 
with older releases of PowerAI Vision as well).



## Who am I ?
I'm [Maxime Deloche](https://www.linkedin.com/in/maximedeloche) and I work as a deep learning engineer in the Cognitive Systems Lab, at the [IBM Systems Center of Montpellier](https://www.ibm.com/ibm/clientcenter/montpellier/index.shtml), France. Our team is providing pre-sales technical support to IBM teams, partners and customers in Europe who are interested in Power Systems based infrastruture for AI: including [IBM Visual Insights, WMLCE, WMLA](https://developer.ibm.com/linuxonpower/deep-learning-powerai/)...

> _For any question or assistance, feel free to contact me or open an issue, and we'll get back to you._


## Documentation
Source: [IBM Knowledge Center - Working with custom models](https://www.ibm.com/support/knowledgecenter/SSRU69_1.1.3/base/vision_work_custom.html)

To import a custom model in Visual Insights, you must build a `zip` file containing at least 2 files:
`train.py` and `deploy.py`

* `train.py`: must define a `MyTrain` class implementing `TrainCallback`
	* template:
	```python
	from train_interface import TrainCallback

	class MyTrain(TrainCallback):
		def __init__(self):
			pass
		def onPreprocessing(self, labels, images, workspace_path, params):
			pass
		def onTraining(self, monitor_handler):
			pass
		def onCompleted(self, model_path):
			pass
		def onFailed(self, train_status, e, tb_message):
			pass
	```
	* procedure:
		* `onPreprocessing` first called
		* then `onTraining`
		* if error raised: `onFailed` called
		* else: `onCompleted` called

* `deploy.py`: must define a `MyDeploy` class implementing `DeployCallback`
	* template:
	```python
	from deploy_interface import DeployCallback

	class MyDeploy(DeployCallback):
		def __init__(self):
			pass
		def onModelLoading(self, model_path, labels, workspace_path):
			pass
		def onTest(self):
			pass
		def onInference(self, image_url, params):
			pass
		def onFailed(self, deploy_status, e, tb_message):
			pass
	```
	* procedure:
		* `onModelLading` first called
		* then `onInference`
		* if error raised: `onFailed` called


> Note: those procedures are implemented in `tests/test_My{Train,Deploy}.py` files, allowing to emulate Visual Insights behavior and therefore test models locally


## Resources

### Import in Visual Insights

To import a model in PAIV, put `train.py` and `deploy.py` in a zip file, and upload that zip file in PAIV, section `Custom Assets`.

> Note that object detection uses code in the SSD7/ directory that you have to include in the zip file as well.

#### Image classification example
```bash
zip ../model.zip train.py deploy.py
```

#### Object detection example
```bash
zip -r ../model.zip train.py deploy.py SSD7/
```


### Content

Both folders (`image_classification/` and `object_detection/`) have the same following structure:

* `src/train.py`: class implementing model training
* `src/deploy.py`: class implementing model inference
* `src/train_interface.py` and `src/deploy_interface.py`: empty classes to emulate PAIV source code, in order for the test to run outside PAIV
* `tests/`: tests that emulate PAIV behavior, ie. call callbacks in appropriate order and handle errors

### Datasets

All datasets are available both as:
* `original` version (zip file containing images and labels)
* `paiv` version (zip file that can be directly imported in the datasets of Visual Insights)

[__Download link__](https://ibm.box.com/s/0ndbfvbz71ar3j0u1r4f4g0xn7rq9p06)


If you want to perform local tests (outside PAIV), download the original dataset and unzip it in `datasets/monkeys-dataset/` or `datasets/bears-dataset` 
(or set a different path in the test files).

#### Monkeys :monkey: (image classification)

Dataset source: [10-monkey-species on Kaggle](https://www.kaggle.com/slothkong/10-monkey-species)

#### Bears :bear: (object detection)

Dataset source: extract from [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)


## Testing

### Run tests

These scripts are meant to simulate the behavior or Visual Insights, allowing to test locally
your code before uploading it to Visual Insights.

* `tests/test_MyTrain.py`: test the training of the model using `src.train.MyTrain` class:
calls `onPreprocessing`, `onTraining`, and `onCompleted` or `onFailed` depending on the training result

```bash
./tests/test_MyTrain.py
```

* `tests/test_MyDeploy.py`: test the inference using `src.deploy.MyDeploy` class: calls `onModelLoading`,
`onInference` on a dataset image and `onFailed` if error raised during deployment or inference

```bash
./tests/test_MyDeploy.py
```

## Debugging

If logs of PAIV instance are not recorded (in Kibana for example), they can be found in Docker logs. As the Docker for the training is
removed as soon as the training ends, in case of failure you don't have enough time to find the docker ID and get its log: a 'dirty'
workaround is to add a `sleep(60)` in the `onFailure` callback ensuring the container stays alive for a minute and giving you time to
launch the `docker logs` function below (it can be removed in prod or if you record the logs).

> Note: PAIV somehow sets the python log level to logging.INFO, no matter what you set in your script, thus you need to write logs in
at least INFO level.

To display logs of the most recent Visual Insights classification training:

```bash
sudo docker logs -f `sudo docker ps -qf "name=k8s_visual-insights-cic-train" | head -n 1`
```

or

```bash
kubectl logs -f `kubectl get pods --sort-by=.status.startTime | tail -n 1 | awk '{print $1}'`
```

The `docker ps` get the IDs of containers containing 'k8s_visual-insights-cic-train' in their name (indicating classification trainings)
sorted by date.  `head` gets the most recent one, and `docker logs` displays its log stack.

