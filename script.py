def accumulate(state, output):
    QUALIFIED_BATCH_SIZE = 20
    current_acc_size = state.shape[0]
    if not bool(state.sum()):
        state = output
    elif current_acc_size < QUALIFIED_BATCH_SIZE:
        state = torch.cat((state, output))
    else:
        state = torch.zeros((1, 1))
        # It's ideal to call drift detector from here
    return state


def push_statistics(state, p_value):
    QUALIFIED_BATCH_SIZE = 20
    current_acc_size = state.shape[0]
    if current_acc_size == QUALIFIED_BATCH_SIZE:
        out = str(float(p_value))
        redis.execute("RPUSH", "statistics", out)
