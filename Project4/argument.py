def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--decay_rate', type=int, default=2e4, help='decay rate')
    parser.add_argument('--target_update_freq', type=int, default=5000, help='target network update frequency')

    # Optimizer choice
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['adam', 'rmsprop'], help='optimizer')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save log files')

    parser.add_argument('--dqn_type', type=str, default='dqn', choices=['double dqn', 'dueling dqn', 'dqn', 'double dueling dqn'], help='choose dqn model')
    
    parser.add_argument('--buffer_type', type=str, default='std_buff', choices=['std_buff', 'prioritized_buff'], help='choose buffer type')
    return parser