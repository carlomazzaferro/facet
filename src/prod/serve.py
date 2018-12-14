import hug


@hug.get('/echo', versions=1)
def echo(text):
    return text
