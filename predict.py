import tools

if __name__ == '__main__':
    args = tools.parse_args_predict()
    model = tools.load_model(args['load_path'])
    print(model.predict(args['input_value']))
