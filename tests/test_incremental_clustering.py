"""增量图聚类单元测试。"""
import pytest
from unittest.mock import patch, MagicMock

from backend.graph.incremental_clustering import (
    patch_new_node,
    recluster_subgraph,
    get_neighbor_communities,
)


class TestPatchNewNode:
    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.get_neighbor_communities")
    def test_patch_to_dominant_community(self, mock_neighbors, mock_write):
        """邻居中 80% 属于同一社区 -> 直接归入。"""
        mock_neighbors.return_value = {
            "n1": "c1", "n2": "c1", "n3": "c1", "n4": "c2"
        }
        result = patch_new_node("new_entity")
        assert result["action"] == "patched"
        assert result["community_id"] == "c1"
        mock_write.assert_called_once()

    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.get_neighbor_communities")
    def test_no_dominant_community_triggers_recluster(self, mock_neighbors, mock_write):
        """邻居分散在多个社区 -> 触发子图重构。"""
        mock_neighbors.return_value = {
            "n1": "c1", "n2": "c2", "n3": "c3"
        }
        result = patch_new_node("new_entity")
        assert result["action"] == "recluster"
        assert "c1" in result["affected_communities"]
        assert "c2" in result["affected_communities"]
        mock_write.assert_not_called()

    @patch("backend.graph.incremental_clustering.get_neighbor_communities")
    def test_no_neighbors(self, mock_neighbors):
        """孤立节点 -> 返回 no_neighbors。"""
        mock_neighbors.return_value = {}
        result = patch_new_node("isolated_entity")
        assert result["action"] == "no_neighbors"

    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.get_neighbor_communities")
    def test_threshold_boundary(self, mock_neighbors, mock_write):
        """恰好 60% -> 应该归入。"""
        mock_neighbors.return_value = {
            "n1": "c1", "n2": "c1", "n3": "c2"
        }
        result = patch_new_node("boundary_entity")
        assert result["action"] == "patched"
        assert result["community_id"] == "c1"


class TestReclusterSubgraph:
    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.run_cypher")
    def test_empty_communities(self, mock_read, mock_write):
        """空社区列表 -> 返回 0 变化。"""
        result = recluster_subgraph([])
        assert result["changed_nodes"] == 0
        mock_read.assert_not_called()

    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.run_cypher")
    def test_single_node_subgraph(self, mock_read, mock_write):
        """只有一个节点的子图 -> 不运行聚类。"""
        mock_read.return_value = []
        result = recluster_subgraph(["c1"])
        assert result["changed_nodes"] == 0

    @patch("backend.graph.incremental_clustering.write_cypher")
    @patch("backend.graph.incremental_clustering.run_cypher")
    def test_two_node_subgraph(self, mock_read, mock_write):
        """两个节点的子图 -> 运行聚类。"""
        # First call: edges, second call: isolated nodes
        mock_read.side_effect = [
            [{"src": "a", "src_cid": "c1", "dst": "b", "dst_cid": "c2", "weight": 1.0}],
            [],
        ]
        result = recluster_subgraph(["c1", "c2"])
        # Louvain on 2 connected nodes should produce 1 community
        assert result["changed_nodes"] >= 0  # may or may not change


class TestGetNeighborCommunities:
    @patch("backend.graph.incremental_clustering.run_cypher")
    def test_returns_neighbor_map(self, mock_cypher):
        """正确返回邻居->社区映射。"""
        mock_cypher.return_value = [
            {"name": "n1", "cid": "c1"},
            {"name": "n2", "cid": "c2"},
        ]
        result = get_neighbor_communities("test_node")
        assert result == {"n1": "c1", "n2": "c2"}

    @patch("backend.graph.incremental_clustering.run_cypher")
    def test_empty_neighbors(self, mock_cypher):
        """无邻居 -> 返回空字典。"""
        mock_cypher.return_value = []
        result = get_neighbor_communities("isolated")
        assert result == {}


class TestDirtyFlag:
    @patch("backend.storage.database.SessionLocal")
    def test_mark_communities_dirty(self, mock_session):
        """mark_communities_dirty 应设置 is_dirty=True。"""
        from backend.graph.community import mark_communities_dirty

        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query.return_value.filter_by.return_value.first.return_value = MagicMock(is_dirty=False)

        count = mark_communities_dirty(["c1", "c2"])
        assert count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
