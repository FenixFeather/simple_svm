#!/usr/bin/env julia

using Distributions

function parse_data(path::String)
    fd = open(path)
    lines = readlines(fd)
    return [([int(number) for number in split(line, ',')[2:end - 1]], ifelse(int(split(line, ',')[end]) == 2, -1, 1)) for line in lines]
    ## return filter(line->!(in("?", line)), stuff)
end

function gradient(a, x, b, y, lambda)
    if y * (dot(a, x) + b) >= 1
        return [lambda * a, 0]
    else
        return [lambda * a - y * x, -y]
    end
end

function current_accuracy(u::Vector, test_stuff)
    w = u[1:end-1]
    b = u[end]
    accuracy_vector = [int((dot(w, vector[1]) + b > 0 ? 1 : -1) == vector[2]) for vector in test_stuff]
    return sum(accuracy_vector)/length(accuracy_vector)
end

function classification_vector(u::Vector, test_stuff)
    w = u[1:end-1]
    b = u[end]
    return [(dot(w, vector[1]) + b > 0 ? 1 : -1) for vector in test_stuff]
end

function main(epochs::Number, steps::Number, lambda::Number, path::String)
    feature_vectors = shuffle(parse_data(path))
    validation_vectors = feature_vectors[1:100]
    test_vectors = feature_vectors[101:200]
    training_vectors = feature_vectors[201:end]

    a = [rand(1:42) for ii in 1:9]
    b = 0
    u = [a,b]
    
    for epoch in 1:epochs
        evaluation = sample(training_vectors, 50)
        for step in 1:steps
            step_length = 1/(2e * epoch + 1e - 1)
            x = sample(training_vectors)
            p_k = gradient(u[1:end-1], x[1], u[end], x[2], lambda)
            u = u - step_length * p_k
            
            if step % 10 == 0
                println("Accuracy: $(current_accuracy(u, evaluation))")
            end
        end
        println("Accuracy on testing set: $(current_accuracy(u, test_vectors))")
    end

    println("Accuracy on validation set: $(current_accuracy(u, validation_vectors))")
    println("Accuracy on testing set: $(current_accuracy(u, test_vectors))")
end
