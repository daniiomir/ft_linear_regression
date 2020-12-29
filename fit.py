import tools

if __name__ == '__main__':
    args = tools.parse_args_fit()
    model = tools.LinearRegression(args['learning_rate'], args['loss'], args['verbose'])
    data, target = tools.read_data(args['data_path'])
    data = tools.standart_scale(data)
    model.fit(data, target)
    tools.save_model(model, args['save_path'])
