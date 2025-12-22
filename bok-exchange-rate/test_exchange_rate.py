"""
환율 정보 수집 기능 테스트 스크립트
"""
import unittest
import os
from unittest.mock import Mock, patch
import pandas as pd
from exchange_rate_fetcher import ExchangeRateFetcher


class TestExchangeRateFetcher(unittest.TestCase):
    """ExchangeRateFetcher 클래스 테스트"""
    
    def setUp(self):
        """테스트 초기화"""
        self.api_key = "test_api_key"
        self.fetcher = ExchangeRateFetcher(self.api_key)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.fetcher.api_key, self.api_key)
        self.assertIn('USD', ExchangeRateFetcher.CURRENCY_CODES)
        self.assertIn('JPY', ExchangeRateFetcher.CURRENCY_CODES)
        self.assertIn('CNY', ExchangeRateFetcher.CURRENCY_CODES)
    
    def test_currency_codes(self):
        """통화 코드 확인"""
        self.assertEqual(ExchangeRateFetcher.CURRENCY_CODES['USD'], '0000001')
        self.assertEqual(ExchangeRateFetcher.CURRENCY_CODES['JPY'], '0000002')
        self.assertEqual(ExchangeRateFetcher.CURRENCY_CODES['CNY'], '0000053')
    
    def test_invalid_currency(self):
        """잘못된 통화 코드 처리 테스트"""
        with self.assertRaises(ValueError):
            self.fetcher.fetch_exchange_rate('INVALID', '20240101', '20240131')
    
    @patch('exchange_rate_fetcher.requests.get')
    def test_fetch_exchange_rate_success(self, mock_get):
        """환율 정보 가져오기 성공 테스트"""
        # Mock 응답 설정
        mock_response = Mock()
        mock_response.json.return_value = {
            'StatisticSearch': {
                'list_total_count': '2',
                'row': [
                    {
                        'TIME': '20240101',
                        'DATA_VALUE': '1300.50',
                        'UNIT_NAME': '원/달러'
                    },
                    {
                        'TIME': '20240102',
                        'DATA_VALUE': '1305.75',
                        'UNIT_NAME': '원/달러'
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # 테스트 실행
        df = self.fetcher.fetch_exchange_rate('USD', '20240101', '20240102')
        
        # 검증
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('datetime', df.columns)
        self.assertIn('DATA_VALUE', df.columns)
        self.assertIn('currency', df.columns)
        self.assertEqual(df['currency'].iloc[0], 'USD')
    
    @patch('exchange_rate_fetcher.requests.get')
    def test_fetch_exchange_rate_api_error(self, mock_get):
        """API 오류 처리 테스트"""
        # Mock 응답 설정 - API 오류
        mock_response = Mock()
        mock_response.json.return_value = {
            'RESULT': {
                'CODE': 'ERR',
                'MESSAGE': 'Invalid API Key'
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # 테스트 실행 및 검증
        with self.assertRaises(Exception):
            self.fetcher.fetch_exchange_rate('USD', '20240101', '20240131')
    
    @patch('exchange_rate_fetcher.requests.get')
    def test_fetch_multiple_rates(self, mock_get):
        """여러 통화 환율 가져오기 테스트"""
        # Mock 응답 설정
        mock_response = Mock()
        mock_response.json.return_value = {
            'StatisticSearch': {
                'list_total_count': '1',
                'row': [
                    {
                        'TIME': '20240101',
                        'DATA_VALUE': '1300.50',
                        'UNIT_NAME': '원'
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # 테스트 실행
        results = self.fetcher.fetch_multiple_rates(
            ['USD', 'JPY'], 
            '20240101', 
            '20240101'
        )
        
        # 검증
        self.assertIsInstance(results, dict)
        self.assertIn('USD', results)
        self.assertIn('JPY', results)
        self.assertIsInstance(results['USD'], pd.DataFrame)
        self.assertIsInstance(results['JPY'], pd.DataFrame)
    
    @patch('exchange_rate_fetcher.requests.get')
    def test_get_latest_rate(self, mock_get):
        """최신 환율 조회 테스트"""
        # Mock 응답 설정
        mock_response = Mock()
        mock_response.json.return_value = {
            'StatisticSearch': {
                'list_total_count': '1',
                'row': [
                    {
                        'TIME': '20241220',
                        'DATA_VALUE': '1450.25',
                        'UNIT_NAME': '원/달러'
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # 테스트 실행
        latest = self.fetcher.get_latest_rate('USD')
        
        # 검증
        self.assertIsNotNone(latest)
        self.assertEqual(latest['currency'], 'USD')
        self.assertEqual(latest['rate'], 1450.25)
        self.assertIn('date', latest)
        self.assertIn('unit', latest)


class TestExchangeRateIntegration(unittest.TestCase):
    """통합 테스트 (실제 API 호출)"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        cls.api_key = os.getenv('BOK_API_KEY')
        cls.skip_integration = cls.api_key is None or cls.api_key == 'YOUR_API_KEY_HERE'
    
    def setUp(self):
        """테스트 초기화"""
        if self.skip_integration:
            self.skipTest("BOK_API_KEY 환경변수가 설정되지 않았습니다.")
    
    def test_real_api_call(self):
        """실제 API 호출 테스트"""
        fetcher = ExchangeRateFetcher(self.api_key)
        
        # 최근 데이터 조회
        df = fetcher.fetch_exchange_rate('USD', '20241201', '20241210')
        
        # 검증
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertTrue(all(df['DATA_VALUE'] > 0))
    
    def test_real_latest_rate(self):
        """실제 최신 환율 조회 테스트"""
        fetcher = ExchangeRateFetcher(self.api_key)
        
        latest = fetcher.get_latest_rate('USD')
        
        # 검증
        self.assertIsNotNone(latest)
        self.assertGreater(latest['rate'], 0)
        self.assertEqual(latest['currency'], 'USD')


def run_basic_tests():
    """기본 테스트 실행"""
    print("=" * 60)
    print("환율 정보 수집 기능 테스트")
    print("=" * 60)
    
    # Mock 테스트 실행
    print("\n[1단계] Mock 테스트 실행")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestExchangeRateFetcher)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 통합 테스트 실행
    print("\n[2단계] 통합 테스트 실행")
    suite = loader.loadTestsFromTestCase(TestExchangeRateIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_basic_tests()
