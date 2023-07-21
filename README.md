# ML pipeline with Sagemaker and CircleCI

This is a demonstration of integrating Sagemaker with CircleCI to create an end-to-end ML pipeline, including model training and deployment. It uses a monorepo setup where each model is contained in its own folder. Furthermore, it uses CircleCI's dynamic configs to adapt each pipeline to the model that experienced code changes.

A walkthrough of the project can be found in [this Medium post](https://medium.com/@ticheung/machine-learning-ci-cd-with-circleci-and-aws-sagemaker-5ff67a0d937f).
