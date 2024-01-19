#!/usr/bin/env python

import asyncio
import sys
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from icmplib import (
    AsyncSocket,
    Host,
    ICMPLibError,
    ICMPRequest,
    ICMPv4Socket,
    ICMPv6Socket,
)
from icmplib.utils import is_ipv6_address, unique_identifier
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QSplitter,
    QTableView,
)


class DataframeTableModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.dataframe = dataframe

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemNeverHasChildren

    def rowCount(self, parent=QModelIndex()):
        return len(self.dataframe) if parent == QModelIndex() else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self.dataframe.columns) if parent == QModelIndex() else 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.dataframe.iloc[index.row(), index.column()])
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self.dataframe.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self.dataframe.index[section])
        return None


class DefaultTableModel(DataframeTableModel):
    nextFrame = Signal(pd.DataFrame)

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(dataframe, parent)
        self._variableDataTopLeftIndex: Optional[QPersistentModelIndex] = None
        self._variableDataBottomRightIndex: Optional[QPersistentModelIndex] = None

    @Slot(list)
    def receiveFrame(self, data: list[dict]):
        dataframe = pd.DataFrame(data)
        self.dataframe.update(dataframe)
        self.nextFrame.emit(self.dataframe.copy(True))
        self.dataChanged.emit(*self._getVariableDataIndexes(dataframe))

    def _getVariableDataIndexes(self, dataframe: pd.DataFrame):
        if not self._variableDataBottomRightIndex:
            top = 0
            bottom = len(dataframe) - 1

            q = filter(lambda x: x != 'address', dataframe.columns)
            indexers = self.dataframe.columns.get_indexer(q)
            left = min(indexers)
            right = max(indexers)

            self._variableDataTopLeftIndex = QPersistentModelIndex(
                self.createIndex(top, left)
            )
            self._variableDataBottomRightIndex = QPersistentModelIndex(
                self.createIndex(bottom, right)
            )

        return self._variableDataTopLeftIndex, self._variableDataBottomRightIndex


class JointTableModel(DataframeTableModel):
    def __init__(self, parent=None):
        super().__init__(pd.DataFrame(), parent)
        self._variableDataTopLeftIndex: Optional[QPersistentModelIndex] = None
        self._variableDataBottomRightIndex: Optional[QPersistentModelIndex] = None

    @Slot(pd.DataFrame)
    def receiveFrame(self, dataframe: pd.DataFrame):
        if not self._variableDataTopLeftIndex:
            self._initDataFrame(dataframe)
        else:
            self._updateDataFrame(dataframe)

    def _createJointDataframe(self, dataframe: pd.DataFrame):
        data = []
        for i in range(0, len(dataframe), 2):
            v4 = dataframe.loc[i]
            v6 = dataframe.loc[i + 1]
            region = v4['region']
            min_rtt = min(v4['min_rtt'], v6['min_rtt'])
            max_rtt = max(v4['max_rtt'], v6['max_rtt'])
            sent = v4['sent'] + v6['sent']
            recv = v4['recv'] + v6['recv']
            avg_rtt = int(
                (v4['avg_rtt'] * v4['recv'] + v6['avg_rtt'] * v6['recv'])
                / (recv + 1e-6)
            )
            loss = int(((sent - recv) / (sent + 1e-6)) * 100)
            jitter = int((v4['jitter'] + v6['jitter']) / 2)
            alive = v4['alive'] or v6['alive']
            data.append(
                {
                    'region': region,
                    'min_rtt': min_rtt,
                    'max_rtt': max_rtt,
                    'sent': sent,
                    'recv': recv,
                    'avg_rtt': avg_rtt,
                    'loss': loss,
                    'jitter': jitter,
                    'alive': alive,
                }
            )
        return pd.DataFrame(data)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            match self.dataframe.columns[index.column()]:
                case 'alive':
                    cast = bool
                case (
                    'min_rtt'
                    | 'avg_rtt'
                    | 'max_rtt'
                    | 'sent'
                    | 'recv'
                    | 'loss'
                    | 'jitter'
                ):
                    cast = int
                case _:
                    cast = str
            return cast(self.dataframe.iloc[index.row(), index.column()])
        return None

    def _initDataFrame(self, dataframe: pd.DataFrame):
        self.beginResetModel()
        self.dataframe = self._createJointDataframe(dataframe)
        self.endResetModel()

        self._variableDataTopLeftIndex = QPersistentModelIndex(self.createIndex(0, 1))
        self._variableDataBottomRightIndex = QPersistentModelIndex(
            self.createIndex(len(self.dataframe) - 1, len(self.dataframe.columns) - 1)
        )

    def _updateDataFrame(self, dataframe: pd.DataFrame):
        self.dataframe.update(self._createJointDataframe(dataframe))
        self.dataChanged.emit(
            self._variableDataTopLeftIndex, self._variableDataBottomRightIndex
        )


class Ping:
    def __init__(self, ip):
        self._address = ip
        self._rtts = []
        self._packets_sent = 0
        self._running = False

    @property
    def address(self):
        return self._address

    async def start(self, interval=1, timeout=2):
        Socket = ICMPv6Socket if is_ipv6_address(self._address) else ICMPv4Socket
        id = unique_identifier()
        with AsyncSocket(Socket(None, False)) as sock:
            seq = 0
            self._running = True
            while self._running:
                await asyncio.sleep(interval)

                request = ICMPRequest(destination=self._address, id=id, sequence=seq)
                seq += 1

                try:
                    sock.send(request)
                    self._packets_sent += 1

                    reply = await sock.receive(request, timeout)
                    reply.raise_for_status()

                    rtt = (reply.time - request.time) * 1000
                    self._rtts.append(rtt)

                except ICMPLibError:
                    pass

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    @property
    def status(self):
        """

        No need of implementing TTL cache as the querying has same interval too.
        """
        return Host(self._address, self._packets_sent, self._rtts)


class MultiPingThread(QThread):
    nextFrame = Signal(list)

    def __init__(
        self,
        ips: Iterable,
        ping_interval=1,
        query_interval=1,
        timeout=2,
        parent=None,
    ):
        super().__init__(parent)
        self._running = False
        self._ips = ips
        self._ping_interval = ping_interval
        self._query_interval = query_interval
        self._timeout = timeout
        self._pingObjs: list[Ping] = []

    def run(self):
        """

        Create a new event loop for this thread as it's not possible to add
        non-gui tasks to the main event loop.
        """
        self._running = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._pingObjs = [Ping(ip) for ip in self._ips]
        # The asyncio.run() shall only be called in the main thread.
        # The other coroutines shall be called by other methods like this.
        # ref: https://docs.python.org/zh-cn/3/library/asyncio-runner.html#asyncio.run
        loop.run_until_complete(self._emitFrames())
        loop.close()

    async def _emitFrames(self):
        loop = asyncio.get_running_loop()
        tasks: list[asyncio.Task] = []
        for ping in self._pingObjs:
            tasks.append(
                loop.create_task(
                    ping.start(self._ping_interval, self._timeout), name=ping.address
                )
            )

        while self._running:
            await asyncio.sleep(self._query_interval)
            data = []
            for pingObj in self._pingObjs:
                status = pingObj.status
                data.append(
                    {
                        'address': pingObj.address,
                        'min_rtt': int(status.min_rtt),
                        'avg_rtt': int(status.avg_rtt),
                        'max_rtt': int(status.max_rtt),
                        'sent': status.packets_sent,
                        'recv': status.packets_received,
                        'loss': int(status.packet_loss * 100),
                        'jitter': int(status.jitter),
                        'alive': status.is_alive,
                    }
                )
            self.nextFrame.emit(data)
        await asyncio.gather(*tasks)

    def stop(self):
        self._running = False
        for ping in self._pingObjs:
            ping.stop()
        self.wait()


class DefaultTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # fmt: off
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionsMovable(True)
        # fmt: on

        self.setAlternatingRowColors(True)
        # TODO: Not working
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)


class MainWindow(QMainWindow):
    def __init__(self, dataframe: pd.DataFrame, config: dict, parent=None):
        super().__init__(parent)
        self.resize(1600, 1000)

        # Models
        jointTableModel = JointTableModel()

        defaultTableModel = DefaultTableModel(dataframe)
        defaultTableModel.nextFrame.connect(jointTableModel.receiveFrame)

        sortProxyModel = QSortFilterProxyModel()
        sortProxyModel.setSourceModel(jointTableModel)

        # Worker
        self._worker = MultiPingThread(
            dataframe['address'],
            config['ping_interval'],
            config['view_interval'],
            config['timeout'],
        )
        self._worker.nextFrame.connect(defaultTableModel.receiveFrame)

        # Views
        defaultTableView = DefaultTableView()
        defaultTableView.setModel(defaultTableModel)

        jointTableView = DefaultTableView()
        jointTableView.setSortingEnabled(True)
        jointTableView.setModel(sortProxyModel)

        splitter = QSplitter()
        splitter.addWidget(defaultTableView)
        splitter.addWidget(jointTableView)
        splitter.setSizes([860, 740])

        self.setCentralWidget(splitter)

        self._worker.start()

    def closeEvent(self, event: QCloseEvent):
        self.close()
        self._worker.stop()
        event.accept()


def load_crawled_data(path: str):
    """

    The adjacent 2 rows (e.g. (0, 1), (2, 3)...) of the output dataframe belong
    to the same host.
    """
    src = pd.read_json(path)

    # Drop unnecessary columns
    src.drop('city_code', axis=1, inplace=True)

    # Split ipv4 and ipv6 columns and merge them into 'address'
    src_v4 = src.copy().drop('ipv6', axis=1)
    src_v4.rename(columns={'ipv4': 'address'}, inplace=True)
    src_v6 = src.copy().drop('ipv4', axis=1)
    src_v6.rename(columns={'ipv6': 'address'}, inplace=True)
    dst_np = np.empty_like(src_v4.values)
    dst_np = np.vstack([dst_np, dst_np])
    dst_np[::2] = src_v4.values
    dst_np[1::2] = src_v6.values
    dst = pd.DataFrame(dst_np, columns=src_v4.columns)

    # Merge `continent`, `country_code` and `city` into `region`
    for i in range(len(dst)):
        continent = dst.loc[i, 'continent']
        country_code = dst.loc[i, 'country_code']
        city = dst.loc[i, 'city']
        dst.loc[i, 'continent'] = f'{continent}-{country_code}-{city}'
    dst.rename(columns={'continent': 'region'}, inplace=True)
    dst.drop(['country_code', 'city'], axis=1, inplace=True)

    # Add new columns
    COLUMNS = {
        'min_rtt': 0,
        'avg_rtt': 0,
        'max_rtt': 0,
        'sent': 0,
        'recv': 0,
        'loss': 0,
        'jitter': 0,
        'alive': False,
    }
    for k, v in COLUMNS.items():
        dst.insert(len(dst.columns), k, v)

    return dst


def main():
    app = QApplication(sys.argv)

    with open('config.yml') as fs:
        config = yaml.safe_load(fs)
    dataframe = load_crawled_data(config['file'])

    mainWindow = MainWindow(dataframe, config)
    mainWindow.show()

    app.exec()


if __name__ == '__main__':
    main()
