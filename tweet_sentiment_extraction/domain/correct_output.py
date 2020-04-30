def left_neighbor(token):
    ...:     if token.nbor(-1).whitespace_ == ' ' or token.i <= 1:
    ...:         return token.text
    ...:     else:
    ...:         return token.text + left_neighbor(token.nbor(-1))

    def right_neighbor(token, doc_length):
    ...:     if token.whitespace_ == ' ' or token.i == doc_length:
    ...:         return token.text
    ...:     else:
    ...:         return token.text + right_neighbor(token.nbor(1))


def get_neighbors_until_space(token):
    text_until_space = token.text
    while token.whitespace_ != ' ':
        token = token.nbor(1)
        text_until_space += token.text
    return text_until_space