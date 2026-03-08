import { InfoPage } from "@/app/components/InfoPage";

export default function Terms() {
  return (
    <InfoPage lang="ko" title="이용약관">
      <p>QuickPDFSum 서비스 이용 시 본 약관이 적용됩니다.</p>
      <h2>서비스 성격</h2>
      <p>AI 기반 PDF 요약 기능을 제공하며, 서비스는 사전 고지 없이 변경·중단될 수 있습니다.</p>
      <h2>이용 제한</h2>
      <ul>
        <li>과도한 요청, 비정상 트래픽은 제한할 수 있습니다.</li>
        <li>불법 문서 업로드는 금지됩니다.</li>
      </ul>
      <h2>면책</h2>
      <p>요약 결과는 참고용이며 법률·재무·의료 판단에 대한 최종 근거로 사용해서는 안 됩니다.</p>
    </InfoPage>
  );
}
