def pytest_addoption(parser):
    parser.addoption("--long", action="store_true", dest="long", default=False, help="enable long tests")
    parser.addoption("--extended", action="store_true", dest="extended", default=False, help="enable extended tests")
    parser.addoption("--no-common", action="store_true", dest="no_common", default=False, help="Disable common tests")


def pytest_configure(config):
    markexprs = []
    if not config.option.long:
        markexprs.append("not long")
    if not config.option.extended:
        markexprs.append("not extended")
    if config.option.no_common:
        markexprs.append("not common")

    setattr(config.option, "markexpr", " and ".join(markexprs))
