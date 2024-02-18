def pytest_addoption(parser):
    parser.addoption('--long', action='store_true', dest="long",
                 default=False, help="enable long tests")

def pytest_configure(config):
    if not config.option.long:
        setattr(config.option, 'markexpr', 'not long')
