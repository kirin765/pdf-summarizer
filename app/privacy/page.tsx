import { InfoPage } from "@/app/components/InfoPage";

export default function Privacy() {
  return (
    <InfoPage lang="ko" title="개인정보 처리방침">
      <p>QuickPDFSum(이하 “서비스”)는 이용자의 개인정보를 중요하게 생각합니다.</p>
      <h2>수집 항목</h2>
      <ul>
        <li>IP 주소, 브라우저 정보</li>
        <li>요청 횟수/요청 시점(요청 제한 및 오남용 방지 목적)</li>
        <li>업로드된 PDF 파일(요약 완료 즉시 삭제)</li>
      </ul>
      <h2>보관</h2>
      <ul>
        <li>PDF 본문은 처리 즉시 삭제됩니다.</li>
        <li>요청 제한 정보는 일 단위로 관리합니다.</li>
      </ul>
      <h2>제3자 제공</h2>
      <p>
        요약 처리 과정에서 OpenAI API로 문서 텍스트가 전달될 수 있으나 광고/추적 목적 사용은 하지 않습니다.
      </p>
    </InfoPage>
  );
}
