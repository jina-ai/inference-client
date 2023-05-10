---
title: XXX Title Test
category: 645b13aedb547006c67ccd0f
---

# Getting Started

The Inference Client is a Python package that provides a simple and flexible way to connect to inference models hosted on Jina AI Cloud. 
In this guide, you will learn how to initialize the client, create models, and connect to models.

## [Initialization](initialization)

To use the Inference Client, you need to import the `Client` class from the `inference_client` package. 
There are several ways to initialize the `Client` object, depending on your use case. 
You can use your Jina AI Cloud authentication token, set environment variables, or log in via the Web UI.

## [Creating a Model](create_model)
Jina AI Cloud provides a user-friendly interface for creating and managing your own AI models. 
You can use this interface to create a new model and manage its details.

## [Connecting to Models](connect_model)
Once you have initialized the `Client` object and created your models on Jina AI Cloud, you can connect to them using the `get_model` method. 
This method returns a `BaseClient` object that provides a high-level interface for interacting with the model. 

By following the steps outlined in this guide, you'll be able to connect to your inference models and start integrating them into your application.

## Next steps

:::::{grid} 2
:gutter: 3

::::{grid-item-card} {octicon}`workflow;1.5em` Initialization
:link: initialization
:link-type: doc

Initialize `Client` using authentication token, env variables or Web UI.
::::

::::{grid-item-card} {octicon}`workflow;1.5em` Creating a Model
:link: create_model
:link-type: doc

Create and manage AI models using Jina AI Cloud.
::::

::::{grid-item-card} {octicon}`workflow;1.5em` Connecting to Models
:link: connect_model
:link-type: doc

Connect to models with `get_model` and use `BaseClient` for model-specific functionality.
::::

:::::


```{toctree}
:hidden:

initialization
create_model
connect_model
```

