#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# fake MonitorHandler imitating Visual Insights interface / displays metrics instead of plotting them

class FakeMonitorHandler():
    def __init__(self):
        pass

    def updateTrainMetrics(self, current_iter, max_iter, loss, epoch):
        print("[monitor_handler - train] iter = %d/%d (epoch %d) | loss = %f" % (current_iter, max_iter, epoch, loss))

    def updateTestMetrics(self, current_iter, accuracy, loss, epoch):
        print("[monitor_handler - test] iter = %d (epoch %d) | loss = %f | accuracy = %f" % (current_iter,  epoch, loss, accuracy))
