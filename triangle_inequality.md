To illustrate why words do not lie in Euclidean space, let's pick three words, *water*, *river*, and *bank*, and calculate the distances between them.

    dist(water, river)  = 0.499
    dist(bank,  water)  = 0.159
    dist(bank,  river)  = 0.272

For the cosine similarity between the GloVe vectors to act as a true metric, the distances between the three words should satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality), in which the distance between any two sides should be less than the distance between the third side. However, in this case, the combined distance between the two sides of the triangle, *bank*-*river* and *bank*-*water*, are not greater than the distance between the third side, *water*-*river*.

    0.159 + 0.272 = 0.431 and 0.431 < 0.499