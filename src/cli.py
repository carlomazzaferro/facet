# -*- coding: utf-8 -*-

"""Console script for facet."""
import os


import click
from src.data.make_dataset import create_datasets
from src.models.predict_model import predict_results
import hug
from src.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.models.train_model import train_clf
from src import prod


__author__ = "Carlo Mazzaferro"
__copyright__ = "Carlo Mazzaferro"


@click.group()
def cli():
    """Racket's CLI. With it, you can perform pretty much all operations you desire
        Shown below are all the possible commands you can use.
        Run ::
            $ racket -h
        To get an overview of the possibilities.
    """


@cli.command()
@click.option('--raw', required=True)
@click.option('--train', required=True)
@click.option('--test', required=True)
def make_dataset(raw, train, test):
    r, tr, te = os.path.join(RAW_DATA_PATH, raw), \
                       os.path.join(PROCESSED_DATA_PATH, train), \
                       os.path.join(PROCESSED_DATA_PATH, test)
    create_datasets(r, tr, te)


@cli.command()
@click.option('--file', required=True)
def train(file):
    train_clf(file)


@cli.command()
@click.option('--file', required=True)
def predict(file):
    predict_results(test_file=file)


@cli.command()
def serve():
    hug.API(prod).http.serve()


cli.add_command(make_dataset)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(serve)

