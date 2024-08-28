def get_model(args, rank=None, world_size=None):
    checkpoint = None
    if not args.resume == '':
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.model = checkpoint['args'].model
            args_backup = args
            args = checkpoint['args']
            args.optim = args_backup.optim
            args.momentum = args_backup.momentum
            args.weight_decay = args_backup.weight_decay
            args.dropout = args_backup.dropout
            args.no_batch_norm = args_backup.no_batch_norm
            args.cutout = args_backup.cutout
            args.length = args_backup.length
            print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('Checkpoint not found: {}'.format(args.resume))

    if args.model == 'mlp':
        model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
    elif args.model.startswith('vgg'):
        model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
    elif args.model == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_ch, args.feat_mult, input_dim)
    elif args.model == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim)
    elif args.model == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim)
    elif args.model == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_ch, args.feat_mult, input_dim)
    elif args.model == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_ch, args.feat_mult, input_dim)
    elif args.model == 'wresnet10-8':
        model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet10-8a':
        model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, True)
    elif args.model == 'wresnet16-4':
        model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet16-4a':
        model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, True)
    elif args.model == 'wresnet16-8':
        model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet16-8a':
        model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, True)
    elif args.model == 'wresnet28-10':
        model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet28-10a':
        model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, True)
    elif args.model == 'wresnet40-10':
        model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet40-10a':
        model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, True)
    else:
        print('No valid model defined')

    # Check if to load model
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        args = args_backup

    if args.cuda:
        model.cuda()

    if args.progress_bar:
        from tqdm import tqdm

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam' or args.optim == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               amsgrad=args.optim == 'amsgrad')
    else:
        print('Unknown optimizer')

    model.set_learning_rate(args.lr)
    print(model)
    print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))
