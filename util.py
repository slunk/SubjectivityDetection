def words():
    """
    Return a dictionary from sentence ids to a list of words in that sentence.
    """
    return to_many("data/blog2008/train/words.txt")

def sentences():
    with open("data/blog2008/train/text.txt") as f:
        return {line.split()[0]: ' '.join(line.split()[1:]) for line in f}

def docbias():
    """
    Return a dictionary from document ids to the document's bias.
    """
    return to_one("data/blog2008/train/docbias.txt")

def indoc():
    """
    Return a dictionary from sentence ids to document ids.
    """
    return to_one("data/blog2008/train/doc.txt")

def to_one(fname):
    with open(fname) as f:
        return {line.split()[0]: line.split()[1] for line in f}

def to_many(fname):
    d = {}
    with open(fname) as f:
        for line in f:
            d.setdefault(line.split()[0], []).append(line.split()[1])
    return d
